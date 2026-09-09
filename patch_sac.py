import re
import sys

def patch():
    with open('gap_rl/algorithms/scripts/custom_sac.py', 'r') as f:
        content = f.read()

    # 1. Insert TokenSelfAttention before CustomActor
    attention_code = """
class TokenSelfAttention(nn.Module):
    def __init__(self, d_attn: int = 384, use_attn_lstm: bool = False):
        super().__init__()
        self.use_attn_lstm = use_attn_lstm
        self.d_attn = d_attn
        
        if self.use_attn_lstm:
            # Linear projections for each of the 5 tokens
            self.proj_pn = nn.Linear(256, d_attn)
            self.proj_tcp = nn.Linear(6, d_attn)
            self.proj_gripper = nn.Linear(2, d_attn)
            self.proj_action = nn.Linear(7, d_attn)
            self.proj_grasp = nn.Linear(5, d_attn)
            
            # Q, K, V projections for the self-attention
            self.W_q = nn.Linear(d_attn, d_attn)
            self.W_k = nn.Linear(d_attn, d_attn)
            self.W_v = nn.Linear(d_attn, d_attn)

    def forward(self, obs_t, features_extractor_out):
        if not self.use_attn_lstm:
            return features_extractor_out
            
        pn_feature = features_extractor_out[:, -256:] # (batch, 256)
        tcp_pose = obs_t["tcp_pose"] # (batch, 6)
        gripper_pos = obs_t["gripper_pos"] # (batch, 2)
        action = obs_t["action"] # (batch, 7)
        grasp_exist = obs_t["grasp_exist"] # (batch, 5)
        
        t1 = self.proj_pn(pn_feature) # (batch, d_attn)
        t2 = self.proj_tcp(tcp_pose)
        t3 = self.proj_gripper(gripper_pos)
        t4 = self.proj_action(action)
        t5 = self.proj_grasp(grasp_exist)
        
        tokens = th.stack([t1, t2, t3, t4, t5], dim=1) # (batch, 5, d_attn)
        
        Q = self.W_q(tokens) # (batch, 5, d_attn)
        K = self.W_k(tokens) # (batch, 5, d_attn)
        V = self.W_v(tokens) # (batch, 5, d_attn)
        
        scores = th.bmm(Q, K.transpose(1, 2)) / (self.d_attn ** 0.5) # (batch, 5, 5)
        attn_weights = F.softmax(scores, dim=-1)
        attended_tokens = th.bmm(attn_weights, V) # (batch, 5, d_attn)
        
        flattened = attended_tokens.view(attended_tokens.size(0), -1)
        return flattened

# ============================================================================
# 2. Actor -- per-frame extraction + real windowed LSTM
"""
    content = content.replace("# ============================================================================\n# 2. Actor -- per-frame extraction + real windowed LSTM", attention_code)

    # 2. CustomActor init
    old_actor_init = """        clip_mean: float = 2.0,
        normalize_images: bool = True,
        extra_pred_dim: int = 7,
    ):
"""
    new_actor_init = """        clip_mean: float = 2.0,
        normalize_images: bool = True,
        extra_pred_dim: int = 7,
        use_attn_lstm: bool = False,
        d_attn: int = 384,
    ):
"""
    content = content.replace(old_actor_init, new_actor_init)

    # 3. CustomActor lstm
    old_actor_lstm = """        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
        self.lstm = nn.LSTM(features_dim, features_dim, batch_first=True)
        self.extra_pred = nn.Linear(last_layer_dim, extra_pred_dim)"""
    new_actor_lstm = """        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
        self.use_attn_lstm = use_attn_lstm
        self.token_attn = TokenSelfAttention(d_attn=d_attn, use_attn_lstm=use_attn_lstm)
        lstm_input_dim = 5 * d_attn if use_attn_lstm else features_dim
        self.lstm = nn.LSTM(lstm_input_dim, features_dim, batch_first=True)
        self.extra_pred = nn.Linear(last_layer_dim, extra_pred_dim)"""
    content = content.replace(old_actor_lstm, new_actor_lstm)

    # 4. CustomActor _extract_windowed_features
    old_actor_ext = """            obs_t = preprocess_obs(obs_t, self.orig_observation_space, normalize_images=self.normalize_images)
            frame_feats.append(self.features_extractor(obs_t))
        seq = th.stack(frame_feats, dim=1)  # (batch, n_stack, features_dim)"""
    new_actor_ext = """            obs_t_pre = preprocess_obs(obs_t, self.orig_observation_space, normalize_images=self.normalize_images)
            feat = self.features_extractor(obs_t_pre)
            feat = self.token_attn(obs_t_pre, feat)
            frame_feats.append(feat)
        seq = th.stack(frame_feats, dim=1)  # (batch, n_stack, features_dim)"""
    content = content.replace(old_actor_ext, new_actor_ext)

    # 5. CustomContinuousCritic init
    old_critic_init = """        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        extra_pred_dim: int = 7,
    ):"""
    new_critic_init = """        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        extra_pred_dim: int = 7,
        use_attn_lstm: bool = False,
        d_attn: int = 384,
    ):"""
    content = content.replace(old_critic_init, new_critic_init)

    # 6. CustomContinuousCritic lstm
    old_critic_lstm = """        self.lstm = nn.LSTM(features_dim, features_dim, batch_first=True)

        self.extra_pred_dim = extra_pred_dim"""
    new_critic_lstm = """        self.use_attn_lstm = use_attn_lstm
        self.token_attn = TokenSelfAttention(d_attn=d_attn, use_attn_lstm=use_attn_lstm)
        lstm_input_dim = 5 * d_attn if use_attn_lstm else features_dim
        self.lstm = nn.LSTM(lstm_input_dim, features_dim, batch_first=True)

        self.extra_pred_dim = extra_pred_dim"""
    content = content.replace(old_critic_lstm, new_critic_lstm)

    # 7. CustomContinuousCritic _extract_windowed_features
    old_critic_ext = """                obs_t = preprocess_obs(obs_t, self.orig_observation_space, normalize_images=self.normalize_images)
                frame_feats.append(self.features_extractor(obs_t))
        seq = th.stack(frame_feats, dim=1)"""
    new_critic_ext = """                obs_t_pre = preprocess_obs(obs_t, self.orig_observation_space, normalize_images=self.normalize_images)
                feat = self.features_extractor(obs_t_pre)
                feat = self.token_attn(obs_t_pre, feat)
                frame_feats.append(feat)
        seq = th.stack(frame_feats, dim=1)"""
    content = content.replace(old_critic_ext, new_critic_ext)
    
    # 8. CustomSACPolicy init
    old_policy_init = """        n_critics: int = 2,
        share_features_extractor: bool = False,
        extra_pred_dim: int = 7,
    ):
        self.orig_observation_space = orig_observation_space"""
    new_policy_init = """        n_critics: int = 2,
        share_features_extractor: bool = False,
        extra_pred_dim: int = 7,
        use_attn_lstm: bool = False,
        d_attn: int = 384,
    ):
        self.use_attn_lstm = use_attn_lstm
        self.d_attn = d_attn
        self.orig_observation_space = orig_observation_space"""
    content = content.replace(old_policy_init, new_policy_init)
    
    # 9. CustomSACPolicy make_actor
    old_policy_actor = """                stack_keys=self.stack_keys,
                orig_observation_space=self.orig_observation_space,
            )
        )"""
    new_policy_actor = """                stack_keys=self.stack_keys,
                orig_observation_space=self.orig_observation_space,
                use_attn_lstm=self.use_attn_lstm,
                d_attn=self.d_attn,
            )
        )"""
    content = content.replace(old_policy_actor, new_policy_actor)
    
    # 10. CustomSACPolicy make_critic
    old_policy_critic = """                stack_keys=self.stack_keys,
                orig_observation_space=self.orig_observation_space,
            )
        )"""
    content = content.replace(old_policy_critic, new_policy_actor) # It's exactly the same block

    with open('gap_rl/algorithms/scripts/custom_sac.py', 'w') as f:
        f.write(content)

patch()
print("Patched custom_sac.py")
