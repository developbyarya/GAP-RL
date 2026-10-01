import re

with open("gap_rl/algorithms/scripts/custom_sac.py", "r") as f:
    content = f.read()

# 1. CustomActor init
actor_init_target = """        self.use_lstm = use_lstm
        lstm_input_dim = features_dim
        self.lstm = nn.LSTM(lstm_input_dim, features_dim, batch_first=True)
        self.extra_pred = nn.Linear(last_layer_dim, extra_pred_dim)"""

actor_init_replacement = """        self.use_lstm = use_lstm
        # Unshared PointNet Trunk
        import copy
        self.aux_features_extractor = copy.deepcopy(self.features_extractor)
        self.aux_lstm = nn.LSTM(features_dim, features_dim, batch_first=True)
        
        lstm_input_dim = features_dim * 2
        self.lstm = nn.LSTM(lstm_input_dim, features_dim, batch_first=True)
        self.extra_pred = nn.Linear(features_dim, extra_pred_dim)"""

content = content.replace(actor_init_target, actor_init_replacement)


# 2. CustomActor _extract_windowed_features
actor_extract_target = """        flat_obs_pre = preprocess_obs(flat_obs, self.orig_observation_space, normalize_images=self.normalize_images)
        flat_feat = self.features_extractor(flat_obs_pre)
        
        seq = flat_feat.view(batch_size, n_stack, -1)
        lstm_out, _ = self.lstm(seq)
        return lstm_out[:, -1, :]"""

actor_extract_replacement = """        flat_obs_pre = preprocess_obs(flat_obs, self.orig_observation_space, normalize_images=self.normalize_images)
        
        # Aux Trunk
        flat_feat_aux = self.aux_features_extractor(flat_obs_pre)
        seq_aux = flat_feat_aux.view(batch_size, n_stack, -1)
        aux_lstm_out, _ = self.aux_lstm(seq_aux)
        self._latest_aux_out = aux_lstm_out[:, -1, :]
        
        # RL Trunk
        flat_feat = self.features_extractor(flat_obs_pre)
        seq_rl = flat_feat.view(batch_size, n_stack, -1)
        
        # Late-stage feature concatenation (RL uses Aux features but detached!)
        combined_seq = th.cat([seq_rl, seq_aux.detach()], dim=-1)
        lstm_out, _ = self.lstm(combined_seq)
        return lstm_out[:, -1, :]"""

content = content.replace(actor_extract_target, actor_extract_replacement)


# 3. CustomActor forward methods
actor_fwd_target1 = """            return mean_actions, log_std, kwargs
        # Predict extra aux tasks
        if self.extra_pred_dim == 7:
            extra_pred = self.extra_pred(latent_pi)"""

actor_fwd_replacement1 = """            return mean_actions, log_std, kwargs
        # Predict extra aux tasks
        if self.extra_pred_dim == 7:
            extra_pred = self.extra_pred(self._latest_aux_out)"""

content = content.replace(actor_fwd_target1, actor_fwd_replacement1)

actor_fwd_target2 = """            raise NotImplementedError
        target_pred = self.target_pred(kwargs["latent_sde"])"""

actor_fwd_replacement2 = """            raise NotImplementedError
        target_pred = self.target_pred(self._latest_aux_out)"""

content = content.replace(actor_fwd_target2, actor_fwd_replacement2)

actor_fwd_target3 = """        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs), extra_pred, target_pred"""
actor_fwd_replacement3 = """        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs), extra_pred, target_pred"""


# 4. CustomContinuousCritic init
critic_init_target = """        self.use_lstm = use_lstm
        lstm_input_dim = features_dim
        self.lstm = nn.LSTM(lstm_input_dim, features_dim, batch_first=True)

        self.extra_pred_dim = extra_pred_dim
        self.extra_pred = nn.Linear(features_dim, extra_pred_dim)"""

critic_init_replacement = """        self.use_lstm = use_lstm
        import copy
        self.aux_features_extractor = copy.deepcopy(self.features_extractor)
        self.aux_lstm = nn.LSTM(features_dim, features_dim, batch_first=True)

        lstm_input_dim = features_dim * 2
        self.lstm = nn.LSTM(lstm_input_dim, features_dim, batch_first=True)

        self.extra_pred_dim = extra_pred_dim
        self.extra_pred = nn.Linear(features_dim, extra_pred_dim)"""

content = content.replace(critic_init_target, critic_init_replacement)

# 5. CustomContinuousCritic _extract_windowed_features
# It's the exact same target string, but it appears twice in the file (once for actor, once for critic)
# Since we already replaced the actor one, let's just replace all occurrences!
# Actually, the string occurs twice. Let's make sure it replaces the remaining one.
content = content.replace(actor_extract_target, actor_extract_replacement)

# 6. CustomContinuousCritic forward
critic_fwd_target1 = """        qvalue_input = th.cat([features, actions], dim=-1)
        qvalue_outputs = tuple(q_net(qvalue_input) for q_net in self.q_networks)

        extra_pred = self.extra_pred(features)"""

critic_fwd_replacement1 = """        qvalue_input = th.cat([features, actions], dim=-1)
        qvalue_outputs = tuple(q_net(qvalue_input) for q_net in self.q_networks)

        extra_pred = self.extra_pred(self._latest_aux_out)"""

content = content.replace(critic_fwd_target1, critic_fwd_replacement1)

critic_fwd_target2 = """            raise NotImplementedError
        target_pred = self.target_pred(features)"""

critic_fwd_replacement2 = """            raise NotImplementedError
        target_pred = self.target_pred(self._latest_aux_out)"""

content = content.replace(critic_fwd_target2, critic_fwd_replacement2)

# Verify replacement counts
assert "aux_features_extractor = copy.deepcopy" in content, "Replacement 1 failed"
assert "seq_aux = flat_feat_aux.view" in content, "Replacement 2 failed"
assert "self.extra_pred(self._latest_aux_out)" in content, "Replacement 3 failed"

with open("gap_rl/algorithms/scripts/custom_sac.py", "w") as f:
    f.write(content)
