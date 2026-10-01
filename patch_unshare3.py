import re

with open("gap_rl/algorithms/scripts/custom_sac.py", "r") as f:
    content = f.read()

# 1. CustomActor init
content = content.replace(
    "self.lstm = nn.LSTM(lstm_input_dim, features_dim, batch_first=True)\n        self.extra_pred = nn.Linear(last_layer_dim, extra_pred_dim)",
    """import copy
        self.aux_features_extractor = copy.deepcopy(features_extractor)
        self.aux_lstm = nn.LSTM(features_dim, features_dim, batch_first=True)
        self.lstm = nn.LSTM(features_dim * 2, features_dim, batch_first=True)
        self.extra_pred = nn.Linear(features_dim, extra_pred_dim)"""
)

# 2. CustomContinuousCritic init
content = content.replace(
    "self.lstm = nn.LSTM(lstm_input_dim, features_dim, batch_first=True)\n\n        self.extra_pred_dim = extra_pred_dim",
    """import copy
        self.aux_features_extractor = copy.deepcopy(features_extractor)
        self.aux_lstm = nn.LSTM(features_dim, features_dim, batch_first=True)
        self.lstm = nn.LSTM(features_dim * 2, features_dim, batch_first=True)

        self.extra_pred_dim = extra_pred_dim"""
)

# 3. _extract_windowed_features (appears TWICE)
old_extract = """        flat_obs_pre = preprocess_obs(flat_obs, self.orig_observation_space, normalize_images=self.normalize_images)
        flat_feat = self.features_extractor(flat_obs_pre)
        
        seq = flat_feat.view(batch_size, n_stack, -1)
        lstm_out, _ = self.lstm(seq)
        return lstm_out[:, -1, :]"""

new_extract = """        flat_obs_pre = preprocess_obs(flat_obs, self.orig_observation_space, normalize_images=self.normalize_images)
        
        # Aux Trunk
        flat_feat_aux = self.aux_features_extractor(flat_obs_pre)
        seq_aux = flat_feat_aux.view(batch_size, n_stack, -1)
        aux_lstm_out, _ = self.aux_lstm(seq_aux)
        self._latest_aux_out = aux_lstm_out[:, -1, :]
        
        # RL Trunk
        flat_feat = self.features_extractor(flat_obs_pre)
        seq_rl = flat_feat.view(batch_size, n_stack, -1)
        
        # Combine
        import torch as th
        combined_seq = th.cat([seq_rl, seq_aux.detach()], dim=-1)
        lstm_out, _ = self.lstm(combined_seq)
        return lstm_out[:, -1, :]"""

content = content.replace(old_extract, new_extract)

# 4. Forward passes
content = content.replace('extra_pred = self.extra_pred(kwargs["latent_sde"])', 'extra_pred = self.extra_pred(self._latest_aux_out)')
content = content.replace('target_pred = self.target_pred(kwargs["latent_sde"])', 'target_pred = self.target_pred(self._latest_aux_out)')

content = content.replace('extra_pred = self.extra_pred(features)', 'extra_pred = self.extra_pred(self._latest_aux_out)')
content = content.replace('target_pred = self.target_pred(features)', 'target_pred = self.target_pred(self._latest_aux_out)')

with open("gap_rl/algorithms/scripts/custom_sac.py", "w") as f:
    f.write(content)
