import re

with open("gap_rl/algorithms/scripts/custom_sac.py", "r") as f:
    content = f.read()

# For CustomActor:
# 1. Add aux_features_extractor and aux_lstm
actor_init_target = """        self.use_lstm = use_lstm
        lstm_input_dim = features_dim
        self.lstm = nn.LSTM(lstm_input_dim, features_dim, batch_first=True)"""

actor_init_replacement = """        self.use_lstm = use_lstm
        lstm_input_dim = features_dim * 2  # RL feat + Aux feat detached
        self.lstm = nn.LSTM(lstm_input_dim, features_dim, batch_first=True)
        import copy
        self.aux_features_extractor = copy.deepcopy(self.features_extractor)
        self.aux_lstm = nn.LSTM(features_dim, features_dim, batch_first=True)"""

content = content.replace(actor_init_target, actor_init_replacement)

# 2. Modify CustomActor._extract_windowed_features
# We need to compute both features
actor_extract_target = """            for i in range(self.n_stack):
                unstacked_obs = {
                    k: obs[k][:, i, ...] if k in self.stack_keys else obs[k]
                    for k in obs.keys()
                }
                features.append(self.features_extractor(unstacked_obs))
            
            features = th.stack(features, dim=1)  # (batch, n_stack, features_dim)
            if self.use_lstm:
                # LSTM over the sequence
                lstm_out, (h_n, c_n) = self.lstm(features)
                features = lstm_out[:, -1, :]  # Take the last hidden state
            else:
                # Just take the latest frame's features
                features = features[:, -1, :]"""

actor_extract_replacement = """            aux_features = []
            for i in range(self.n_stack):
                unstacked_obs = {
                    k: obs[k][:, i, ...] if k in self.stack_keys else obs[k]
                    for k in obs.keys()
                }
                features.append(self.features_extractor(unstacked_obs))
                aux_features.append(self.aux_features_extractor(unstacked_obs))
            
            features = th.stack(features, dim=1)  # (batch, n_stack, features_dim)
            aux_features = th.stack(aux_features, dim=1)
            
            if self.use_lstm:
                # Aux LSTM computes purely aux features
                aux_lstm_out, _ = self.aux_lstm(aux_features)
                self._latest_aux_out = aux_lstm_out[:, -1, :]
                
                # RL LSTM takes RL features + detached Aux features
                combined_features = th.cat([features, aux_features.detach()], dim=-1)
                lstm_out, _ = self.lstm(combined_features)
                features = lstm_out[:, -1, :]
            else:
                self._latest_aux_out = aux_features[:, -1, :]
                features = th.cat([features[:, -1, :], aux_features[:, -1, :].detach()], dim=-1)"""

content = content.replace(actor_extract_target, actor_extract_replacement)

# 3. Modify CustomActor.forward to pass aux_out to extra_pred
actor_forward_target = """        latent_pi, latent_sde = self._get_latent(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, log_std, kwargs
        # Predict extra aux tasks
        if self.extra_pred_dim == 7:
            extra_pred = self.extra_pred(latent_pi)"""

actor_forward_replacement = """        latent_pi, latent_sde = self._get_latent(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, log_std, kwargs
        # Predict extra aux tasks using the aux features!
        if self.extra_pred_dim == 7:
            extra_pred = self.extra_pred(self._latest_aux_out)"""

content = content.replace(actor_forward_target, actor_forward_replacement)

# Also need to fix target_pred!
actor_forward_target2 = """            raise NotImplementedError
        target_pred = self.target_pred(kwargs["latent_sde"])"""

actor_forward_replacement2 = """            raise NotImplementedError
        target_pred = self.target_pred(self._latest_aux_out)"""

content = content.replace(actor_forward_target2, actor_forward_replacement2)

# For CustomContinuousCritic:
critic_init_target = """        self.use_lstm = use_lstm
        lstm_input_dim = features_dim
        self.lstm = nn.LSTM(lstm_input_dim, features_dim, batch_first=True)"""

critic_init_replacement = """        self.use_lstm = use_lstm
        lstm_input_dim = features_dim * 2
        self.lstm = nn.LSTM(lstm_input_dim, features_dim, batch_first=True)
        import copy
        self.aux_features_extractor = copy.deepcopy(self.features_extractor)
        self.aux_lstm = nn.LSTM(features_dim, features_dim, batch_first=True)"""

content = content.replace(critic_init_target, critic_init_replacement)

critic_extract_target = """            for i in range(self.n_stack):
                unstacked_obs = {
                    k: obs[k][:, i, ...] if k in self.stack_keys else obs[k]
                    for k in obs.keys()
                }
                features.append(self.features_extractor(unstacked_obs))
            
            features = th.stack(features, dim=1)  # (batch, n_stack, features_dim)
            if self.use_lstm:
                # LSTM over the sequence
                lstm_out, (h_n, c_n) = self.lstm(features)
                features = lstm_out[:, -1, :]  # Take the last hidden state
            else:
                # Just take the latest frame's features
                features = features[:, -1, :]"""

critic_extract_replacement = """            aux_features = []
            for i in range(self.n_stack):
                unstacked_obs = {
                    k: obs[k][:, i, ...] if k in self.stack_keys else obs[k]
                    for k in obs.keys()
                }
                features.append(self.features_extractor(unstacked_obs))
                aux_features.append(self.aux_features_extractor(unstacked_obs))
            
            features = th.stack(features, dim=1)  # (batch, n_stack, features_dim)
            aux_features = th.stack(aux_features, dim=1)
            
            if self.use_lstm:
                aux_lstm_out, _ = self.aux_lstm(aux_features)
                self._latest_aux_out = aux_lstm_out[:, -1, :]
                
                combined_features = th.cat([features, aux_features.detach()], dim=-1)
                lstm_out, _ = self.lstm(combined_features)
                features = lstm_out[:, -1, :]
            else:
                self._latest_aux_out = aux_features[:, -1, :]
                features = th.cat([features[:, -1, :], aux_features[:, -1, :].detach()], dim=-1)"""

content = content.replace(critic_extract_target, critic_extract_replacement)

critic_forward_target = """        qvalue_input = th.cat([features, actions], dim=-1)
        qvalue_outputs = tuple(q_net(qvalue_input) for q_net in self.q_networks)

        extra_pred = self.extra_pred(features)"""

critic_forward_replacement = """        qvalue_input = th.cat([features, actions], dim=-1)
        qvalue_outputs = tuple(q_net(qvalue_input) for q_net in self.q_networks)

        extra_pred = self.extra_pred(self._latest_aux_out)"""

content = content.replace(critic_forward_target, critic_forward_replacement)

critic_forward_target2 = """        if self.extra_pred_dim == 7:
            pass"""

critic_forward_replacement2 = """        if self.extra_pred_dim == 7:
            pass"""
# Wait, critic forward target_pred
critic_forward_target3 = """            )
        target_pred = self.target_pred(features)"""
critic_forward_replacement3 = """            )
        target_pred = self.target_pred(self._latest_aux_out)"""

content = content.replace(critic_forward_target3, critic_forward_replacement3)

with open("gap_rl/algorithms/scripts/custom_sac.py", "w") as f:
    f.write(content)
