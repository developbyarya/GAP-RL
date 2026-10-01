import re
import sys

def remove_from_sac_train():
    with open('gap_rl/algorithms/scripts/sac_train.py', 'r') as f:
        content = f.read()

    # Imports
    content = re.sub(r'from custom_sac import CustomSAC, FrameStackWrapper, FrameStackObsWrapper, default_stack_keys\n', 'from custom_sac import CustomSAC\n', content)

    # Configs
    content = re.sub(r'    n_stack = cfg\.get\("n_stack", 4 if is_goal_aux else 1\)\n    cfg\["n_stack"\] = n_stack\n', '', content)

    # FrameStackWrapper usage
    fs_usage = """    stack_keys = default_stack_keys(orig_obs_space)
    if is_goal_aux:
        # Always wrap CustomSAC envs so stacked keys have a leading time axis
        # (n_stack==1 still yields shape (1, *orig) for a consistent LSTM window).
        vec_env = FrameStackWrapper(vec_env, n_stack=n_stack, stack_keys=stack_keys)
        print(f"FrameStack: n_stack={n_stack}, stack_keys={stack_keys}")"""
    content = content.replace(fs_usage, "")

    # kwargs stack_keys
    content = content.replace("                stack_keys=stack_keys,\n", "")

    # FrameStackObsWrapper usage
    fs_eval_usage = """    if is_goal_aux and n_stack > 1:
        evalenv = FrameStackObsWrapper(evalenv, n_stack=n_stack, stack_keys=stack_keys)"""
    content = content.replace(fs_eval_usage, "")

    with open('gap_rl/algorithms/scripts/sac_train.py', 'w') as f:
        f.write(content)

def remove_from_custom_sac():
    with open('gap_rl/algorithms/scripts/custom_sac.py', 'r') as f:
        content = f.read()
    
    # 1. Remove classes and functions from start until CustomActor
    # The block starts at AUX_OBS_KEYS and goes to class CustomActor
    match = re.search(r'# Aux supervision targets.*?# ============================================================================\n# 2\. Actor', content, re.DOTALL)
    if match:
        content = content.replace(match.group(0), '# ============================================================================\n# 2. Actor')

    # 2. CustomActor init
    content = content.replace("        features_dim: int,\n        stack_keys: List[str],", "        features_dim: int,")
    content = content.replace("        self.stack_keys = stack_keys\n", "")

    # 3. CustomActor _extract_windowed_features
    old_ext = """    def _extract_windowed_features(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        \"\"\"
        obs[key] for key in stack_keys has shape (batch, n_stack, *orig_shape).
        Slice each timestep, run the per-frame features_extractor, LSTM over the
        real sequence.
        \"\"\"
        n_stack = obs[self.stack_keys[0]].shape[1]
        frame_feats = []
        for t in range(n_stack):
            obs_t = {k: (v[:, t] if k in self.stack_keys else v) for k, v in obs.items()}
            obs_t_pre = preprocess_obs(obs_t, self.orig_observation_space, normalize_images=self.normalize_images)
            feat = self.features_extractor(obs_t_pre)
            feat = self.token_attn(obs_t_pre, feat)
            frame_feats.append(feat)
        seq = th.stack(frame_feats, dim=1)  # (batch, n_stack, features_dim)
        lstm_out, _ = self.lstm(seq)
        return lstm_out[:, -1, :]"""
    
    new_ext = """    def _extract_windowed_features(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        obs_pre = preprocess_obs(obs, self.orig_observation_space, normalize_images=self.normalize_images)
        feat = self.features_extractor(obs_pre)
        feat = self.token_attn(obs_pre, feat)
        seq = feat.unsqueeze(1)  # (batch, 1, features_dim)
        lstm_out, _ = self.lstm(seq)
        return lstm_out[:, -1, :]"""
    content = content.replace(old_ext, new_ext)

    # 4. CustomContinuousCritic init
    content = content.replace("        features_dim: int,\n        stack_keys: List[str],", "        features_dim: int,")
    content = content.replace("        self.stack_keys = stack_keys\n", "")

    # 5. CustomContinuousCritic _extract_windowed_features
    old_critic_ext = """    def _extract_windowed_features(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        n_stack = obs[self.stack_keys[0]].shape[1]
        frame_feats = []
        with th.set_grad_enabled(not self.share_features_extractor):
            for t in range(n_stack):
                obs_t = {k: (v[:, t] if k in self.stack_keys else v) for k, v in obs.items()}
                obs_t_pre = preprocess_obs(obs_t, self.orig_observation_space, normalize_images=self.normalize_images)
                feat = self.features_extractor(obs_t_pre)
                feat = self.token_attn(obs_t_pre, feat)
                frame_feats.append(feat)
        seq = th.stack(frame_feats, dim=1)
        lstm_out, _ = self.lstm(seq)
        return lstm_out[:, -1, :]"""

    new_critic_ext = """    def _extract_windowed_features(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        with th.set_grad_enabled(not self.share_features_extractor):
            obs_pre = preprocess_obs(obs, self.orig_observation_space, normalize_images=self.normalize_images)
            feat = self.features_extractor(obs_pre)
            feat = self.token_attn(obs_pre, feat)
        seq = feat.unsqueeze(1)
        lstm_out, _ = self.lstm(seq)
        return lstm_out[:, -1, :]"""
    content = content.replace(old_critic_ext, new_critic_ext)

    # 6. CustomSACPolicy init
    content = content.replace("        orig_observation_space: spaces.Dict,\n        stack_keys: List[str],", "        orig_observation_space: spaces.Dict,")
    content = content.replace("        self.stack_keys = stack_keys\n", "")

    # 7. CustomSACPolicy kwargs
    content = content.replace("                stack_keys=self.stack_keys,\n", "")

    with open('gap_rl/algorithms/scripts/custom_sac.py', 'w') as f:
        f.write(content)

remove_from_sac_train()
remove_from_custom_sac()
print("Patched successfully")
