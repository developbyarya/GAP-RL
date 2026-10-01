import re

def patch():
    with open('gap_rl/algorithms/scripts/custom_sac.py', 'r') as f:
        content = f.read()

    # Actor vectorization
    old_actor_ext = """        n_stack = obs[self.stack_keys[0]].shape[1]
        frame_feats = []
        for t in range(n_stack):
            obs_t = {k: (v[:, t] if k in self.stack_keys else v) for k, v in obs.items()}
            obs_t_pre = preprocess_obs(obs_t, self.orig_observation_space, normalize_images=self.normalize_images)
            feat = self.features_extractor(obs_t_pre)
            feat = self.token_attn(obs_t_pre, feat)
            frame_feats.append(feat)
        seq = th.stack(frame_feats, dim=1)  # (batch, n_stack, features_dim)"""
    new_actor_ext = """        n_stack = obs[self.stack_keys[0]].shape[1]
        batch_size = obs[self.stack_keys[0]].shape[0]
        
        flat_obs = {}
        for k, v in obs.items():
            if k in self.stack_keys:
                flat_obs[k] = v.reshape(batch_size * n_stack, *v.shape[2:])
            else:
                flat_obs[k] = v.repeat_interleave(n_stack, dim=0)
                
        flat_obs_pre = preprocess_obs(flat_obs, self.orig_observation_space, normalize_images=self.normalize_images)
        flat_feat = self.features_extractor(flat_obs_pre)
        flat_feat = self.token_attn(flat_obs_pre, flat_feat)
        
        seq = flat_feat.view(batch_size, n_stack, -1)"""
    content = content.replace(old_actor_ext, new_actor_ext)

    # Critic vectorization
    old_critic_ext = """        with th.set_grad_enabled(not self.share_features_extractor):
            for t in range(n_stack):
                obs_t = {k: (v[:, t] if k in self.stack_keys else v) for k, v in obs.items()}
                obs_t_pre = preprocess_obs(obs_t, self.orig_observation_space, normalize_images=self.normalize_images)
                feat = self.features_extractor(obs_t_pre)
                feat = self.token_attn(obs_t_pre, feat)
                frame_feats.append(feat)
        seq = th.stack(frame_feats, dim=1)"""
    new_critic_ext = """        with th.set_grad_enabled(not self.share_features_extractor):
            batch_size = obs[self.stack_keys[0]].shape[0]
            flat_obs = {}
            for k, v in obs.items():
                if k in self.stack_keys:
                    flat_obs[k] = v.reshape(batch_size * n_stack, *v.shape[2:])
                else:
                    flat_obs[k] = v.repeat_interleave(n_stack, dim=0)
                    
            flat_obs_pre = preprocess_obs(flat_obs, self.orig_observation_space, normalize_images=self.normalize_images)
            flat_feat = self.features_extractor(flat_obs_pre)
            flat_feat = self.token_attn(flat_obs_pre, flat_feat)
            
            seq = flat_feat.view(batch_size, n_stack, -1)"""
    content = content.replace(old_critic_ext, new_critic_ext)

    with open('gap_rl/algorithms/scripts/custom_sac.py', 'w') as f:
        f.write(content)

patch()
print("Patched vectorization")
