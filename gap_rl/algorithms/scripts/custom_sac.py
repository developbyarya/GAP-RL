"""
Frame-stacked version of CustomSAC actor/critic.

Uses a rolling observation window (FrameStackWrapper / FrameStackObsWrapper)
so the LSTM receives a real (batch, n_stack, features_dim) sequence instead of
a single zero-state timestep from i.i.d. ReplayBuffer samples.
"""

import io
import pathlib
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces
from torch import nn
from torch.nn import functional as F

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy, BaseModel
from stable_baselines3.common.preprocessing import preprocess_obs, get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor, create_mlp
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.sac.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3.sac.policies import Actor, LOG_STD_MAX, LOG_STD_MIN
import gym

SelfSAC = TypeVar("SelfSAC", bound="SAC")

# Aux supervision targets: current-step only, never stacked into history.
AUX_OBS_KEYS = ("close_grasp_pose_ee", "eval_target")


def default_stack_keys(observation_space: spaces.Dict) -> List[str]:
    """Stack every Dict key except aux supervision targets."""
    return [k for k in observation_space.spaces.keys() if k not in AUX_OBS_KEYS]


# ============================================================================
# 1. Frame-stack wrappers
# ============================================================================
class FrameStackWrapper(VecEnvWrapper):
    """
    Stacks the given Dict-obs key(s) over the last `n_stack` steps into a new
    leading time axis (n_stack, *orig_shape). Keys NOT in `stack_keys` are
    passed through untouched (current step only).
    """

    def __init__(self, venv, n_stack: int, stack_keys: List[str]):
        assert isinstance(venv.observation_space, spaces.Dict), "FrameStackWrapper needs a Dict obs space"
        self.n_stack = n_stack
        self.stack_keys = stack_keys

        new_spaces = {}
        for key, space in venv.observation_space.spaces.items():
            if key in stack_keys:
                low = np.repeat(space.low[None], n_stack, axis=0)
                high = np.repeat(space.high[None], n_stack, axis=0)
                new_spaces[key] = spaces.Box(low=low, high=high, dtype=space.dtype)
            else:
                new_spaces[key] = space
        observation_space = spaces.Dict(new_spaces)

        super().__init__(venv, observation_space=observation_space)
        self.n_envs = venv.num_envs
        self._history = [{k: deque(maxlen=n_stack) for k in stack_keys} for _ in range(self.n_envs)]

    def _init_history(self, env_idx: int, obs: Dict[str, np.ndarray]):
        for k in self.stack_keys:
            self._history[env_idx][k].clear()
            for _ in range(self.n_stack):
                self._history[env_idx][k].append(obs[k])

    def _push(self, env_idx: int, obs: Dict[str, np.ndarray]):
        for k in self.stack_keys:
            self._history[env_idx][k].append(obs[k])

    def _build_obs(self, env_idx: int, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = {}
        for k, v in obs.items():
            out[k] = np.stack(self._history[env_idx][k], axis=0) if k in self.stack_keys else v
        return out

    def reset(self) -> Dict[str, np.ndarray]:
        obs = self.venv.reset()  # gym==0.21 VecEnv.reset() -> obs only
        stacked = []
        for i in range(self.n_envs):
            single = {k: v[i] for k, v in obs.items()}
            self._init_history(i, single)
            stacked.append(self._build_obs(i, single))
        return _stack_dicts(stacked)

    def _build_terminal_obs(self, env_idx: int, terminal_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Window for the true terminal observation o_T: the last (n_stack-1) frames of the
        finished episode followed by o_T. Needed because SB3 stores it as `next_obs`
        (infos[i]["terminal_observation"]) for the final transition of an episode.
        """
        out = {}
        for k, v in terminal_obs.items():
            if k in self.stack_keys:
                frames = list(self._history[env_idx][k])[1:] + [v]
                out[k] = np.stack(frames, axis=0)
            else:
                out[k] = v
        return out

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()  # gym==0.21 4-tuple
        stacked = []
        for i in range(self.n_envs):
            single = {k: v[i] for k, v in obs.items()}
            if dones[i]:
                # VecEnv auto-resets on done: this `obs` is already the fresh
                # reset observation, so re-seed the window rather than push
                terminal_obs = infos[i].get("terminal_observation")
                if terminal_obs is not None:
                    infos[i]["terminal_observation"] = self._build_terminal_obs(i, terminal_obs)
                self._init_history(i, single)
            else:
                self._push(i, single)
            stacked.append(self._build_obs(i, single))
        return _stack_dicts(stacked), rewards, dones, infos


class FrameStackObsWrapper(gym.ObservationWrapper):
    """
    Single-env (gym) counterpart of FrameStackWrapper for eval / RecordEpisode.
    Also intercepts get_obs() so LoG eval's manual observation path is stacked.
    """

    def __init__(self, env, n_stack: int, stack_keys: List[str]):
        assert isinstance(env.observation_space, spaces.Dict), "FrameStackObsWrapper needs a Dict obs space"
        super().__init__(env)
        self.n_stack = n_stack
        self.stack_keys = list(stack_keys)
        self._history = {k: deque(maxlen=n_stack) for k in stack_keys}

        new_spaces = {}
        for key, space in env.observation_space.spaces.items():
            if key in stack_keys:
                low = np.repeat(space.low[None], n_stack, axis=0)
                high = np.repeat(space.high[None], n_stack, axis=0)
                new_spaces[key] = spaces.Box(low=low, high=high, dtype=space.dtype)
            else:
                new_spaces[key] = space
        self.observation_space = spaces.Dict(new_spaces)

    def align_action_key(self, target_key: str):
        """Swaps history, stack_keys, and observation_space between 'action' and 'prev_action'."""
        if target_key not in ("action", "prev_action"):
            return
        other_key = "prev_action" if target_key == "action" else "action"
        if other_key in self.stack_keys:
            self.stack_keys = [target_key if k == other_key else k for k in self.stack_keys]
        if other_key in self._history:
            self._history[target_key] = self._history.pop(other_key)
        new_spaces = {k: v for k, v in self.observation_space.spaces.items() if k != other_key}
        if other_key in self.observation_space.spaces and target_key not in new_spaces:
            new_spaces[target_key] = self.observation_space.spaces[other_key]
        self.observation_space = spaces.Dict(new_spaces)

    def _init_history(self, obs: Dict[str, np.ndarray]):
        if "action" in self.stack_keys and "action" not in obs and "prev_action" in obs:
            obs["action"] = obs["prev_action"]
        elif "prev_action" in self.stack_keys and "prev_action" not in obs and "action" in obs:
            obs["prev_action"] = obs["action"]
        for k in self.stack_keys:
            self._history[k].clear()
            for _ in range(self.n_stack):
                self._history[k].append(obs[k])

    def _push(self, obs: Dict[str, np.ndarray]):
        if "action" in self.stack_keys and "action" not in obs and "prev_action" in obs:
            obs["action"] = obs["prev_action"]
        elif "prev_action" in self.stack_keys and "prev_action" not in obs and "action" in obs:
            obs["prev_action"] = obs["action"]
        for k in self.stack_keys:
            self._history[k].append(obs[k])

    def _build_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = {}
        for k, space in self.observation_space.spaces.items():
            if k in self.stack_keys:
                out[k] = np.stack(self._history[k], axis=0)
            elif k in obs:
                out[k] = obs[k]
            elif k == "action" and "prev_action" in obs:
                out[k] = obs["prev_action"]
            elif k == "prev_action" and "action" in obs:
                out[k] = obs["action"]
        for k, v in obs.items():
            if k not in out and k not in ("action", "prev_action"):
                out[k] = v
        return out

    def observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self._build_obs(observation)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._init_history(obs)
        return self.observation(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._push(obs)
        return self.observation(obs), reward, done, info

    def get_obs(self, *args, **kwargs):
        obs = self.env.get_obs(*args, **kwargs)
        # After reset, LoG eval often rebuilds obs via get_obs; treat as a push
        # so the window reflects the latest grasp/state setup.
        if all(len(self._history[k]) == self.n_stack for k in self.stack_keys):
            self._push(obs)
        else:
            self._init_history(obs)
        return self.observation(obs)


def _stack_dicts(list_of_dicts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = list_of_dicts[0].keys()
    return {k: np.stack([d[k] for d in list_of_dicts], axis=0) for k in keys}


# ============================================================================
# 1b. Per-frame / temporal boundary helpers
#
#   o[t-k+1..t] --(per-frame GGN/State extractor)--> z[t-k+1..t] --(LSTM)--> h_t
#
# The features_extractor only ever sees ONE frame; all temporal reasoning is in
# TemporalEncoder. Actor and Critic share these helpers instead of duplicating them.
# ============================================================================
class TemporalEncoder(nn.Module):
    """
    Windowed recurrent encoder: [batch, time, features] -> h_t [batch, hidden_dim].

    The LSTM state is NOT persistent across calls: every forward re-encodes the whole
    window from a zero state (matches the replay-buffer windows of FrameStackWrapper).
    """

    def __init__(self, features_dim: int, hidden_dim: Optional[int] = None, num_layers: int = 1):
        super().__init__()
        hidden_dim = hidden_dim or features_dim
        self.lstm = nn.LSTM(
            input_size=features_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_dim = hidden_dim

    def forward(self, sequence: th.Tensor) -> th.Tensor:
        """sequence: [batch, time, features] -> representation at the current (last) timestep."""
        lstm_out, _ = self.lstm(sequence)
        return lstm_out[:, -1, :]


def validate_stack_keys(observation_space: spaces.Dict, stack_keys: List[str]) -> None:
    """
    Every per-frame observation (GGN input AND robot state / prev_action / grasp_exist)
    must be temporal. Only AUX_OBS_KEYS (supervision targets) may stay unstacked.
    """
    missing = [
        k for k in observation_space.spaces.keys()
        if k not in AUX_OBS_KEYS and k not in stack_keys
    ]
    if missing:
        raise ValueError(
            f"Observation keys {missing} are consumed per-frame but are not in stack_keys; "
            f"they would be frozen to the current step while the rest of the window moves."
        )


def _prepare_unstacked_obs(obs: Dict[str, th.Tensor], orig_observation_space: spaces.Dict) -> Dict[str, th.Tensor]:
    """Ensures unstacked flat_obs has exactly the keys in orig_observation_space."""
    flat_obs = {}
    for k in orig_observation_space.spaces.keys():
        if k in obs:
            flat_obs[k] = obs[k]
        elif k == "action" and "prev_action" in obs:
            flat_obs["action"] = obs["prev_action"]
        elif k == "prev_action" and "action" in obs:
            flat_obs["prev_action"] = obs["action"]
    return flat_obs


def _ensure_action_aliases(flat_obs: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
    """Ensures both 'action' and 'prev_action' exist in flat_obs if either is present."""
    if "action" in flat_obs and "prev_action" not in flat_obs:
        flat_obs["prev_action"] = flat_obs["action"]
    elif "prev_action" in flat_obs and "action" not in flat_obs:
        flat_obs["action"] = flat_obs["prev_action"]
    return flat_obs


def extract_frame_sequence(
    features_extractor: nn.Module,
    obs: Dict[str, th.Tensor],
    stack_keys: List[str],
    orig_observation_space: spaces.Dict,
    normalize_images: bool,
) -> th.Tensor:
    """
    Stacked obs  (batch, n_stack, *orig_shape)  ->  per-frame features  [batch, n_stack, features_dim].

    Equivalent to running `features_extractor(obs_t)` for t = 0..n_stack-1 and stacking, but
    folds time into the batch axis so the (GGN) extractor is launched once instead of n_stack
    times. Every frame goes through preprocess_obs against the UNSTACKED observation space.
    """
    batch_size, n_stack = obs[stack_keys[0]].shape[:2]

    flat_obs = {}
    for k in orig_observation_space.spaces.keys():
        src_k = k
        if src_k not in obs:
            if src_k == "action" and "prev_action" in obs:
                src_k = "prev_action"
            elif src_k == "prev_action" and "action" in obs:
                src_k = "action"
        if src_k in obs:
            v = obs[src_k]
            if src_k in stack_keys or k in stack_keys:
                flat_obs[k] = v.reshape(batch_size * n_stack, *v.shape[2:])
            else:
                flat_obs[k] = v.repeat_interleave(n_stack, dim=0)

    flat_obs = preprocess_obs(flat_obs, orig_observation_space, normalize_images=normalize_images)
    flat_obs = _ensure_action_aliases(flat_obs)
    flat_feat = features_extractor(flat_obs)  # (batch * n_stack, features_dim)
    return flat_feat.view(batch_size, n_stack, -1)


# ============================================================================
# 2. Actor -- per-frame extraction + real windowed LSTM

# ============================================================================
class CustomActor(Actor):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        stack_keys: Optional[List[str]] = None,
        orig_observation_space: Optional[spaces.Dict] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        extra_pred_dim: int = 7,
        use_lstm: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            clip_mean,
            normalize_images=normalize_images,
        )
        if orig_observation_space is None:
            orig_observation_space = observation_space
        if stack_keys is None:
            stack_keys = []
        self.stack_keys = stack_keys
        self.orig_observation_space = orig_observation_space
        if stack_keys:
            validate_stack_keys(orig_observation_space, stack_keys)

        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
        self.use_lstm = use_lstm
        # Temporal encoder: z[t-k+1..t] -> h_t. hidden_dim == features_dim because
        # SB3's latent_pi MLP was built against `features_dim`.
        self.temporal_encoder = TemporalEncoder(features_dim=features_dim, hidden_dim=features_dim)

        # Auxiliary heads read latent_pi (output of the policy MLP, width = last_layer_dim):
        #   GGN -> LSTM -> h_t -> latent_pi -+-> mu / log_std (actor)
        #                                    +-> extra_pred (grasp-pose goal)
        #                                    +-> target_pred (eval target)
        self.extra_pred = nn.Linear(last_layer_dim, extra_pred_dim)
        nn.init.xavier_uniform_(self.extra_pred.weight, gain=1)
        nn.init.constant_(self.extra_pred.bias, 0)
        self.extra_pred_dim = extra_pred_dim

        self.target_pred = nn.Linear(last_layer_dim, 4)
        nn.init.xavier_uniform_(self.target_pred.weight, gain=1)
        nn.init.constant_(self.target_pred.bias, 0)

    def _extract_sequence(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        """Stacked obs (batch, n_stack, ...) -> per-frame features z: [batch, n_stack, features_dim]."""
        if not self.stack_keys:
            flat_obs = _prepare_unstacked_obs(obs, self.orig_observation_space)
            flat_obs = preprocess_obs(flat_obs, self.orig_observation_space, normalize_images=self.normalize_images)
            flat_obs = _ensure_action_aliases(flat_obs)
            feat = self.features_extractor(flat_obs)
            return feat.unsqueeze(1)
        return extract_frame_sequence(
            self.features_extractor,
            obs,
            self.stack_keys,
            self.orig_observation_space,
            self.normalize_images,
        )

    def _extract_windowed_features(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        """h_t = LSTM(z_{t-k+1}, ..., z_t) if use_lstm else z_t"""
        if not self.stack_keys and not self.use_lstm:
            flat_obs = _prepare_unstacked_obs(obs, self.orig_observation_space)
            flat_obs = preprocess_obs(flat_obs, self.orig_observation_space, normalize_images=self.normalize_images)
            flat_obs = _ensure_action_aliases(flat_obs)
            return self.features_extractor(flat_obs)
        seq = self._extract_sequence(obs)
        if self.use_lstm:
            return self.temporal_encoder(seq)
        return seq[:, -1, :]

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        h_t = self._extract_windowed_features(obs)
        latent_pi = self.latent_pi(h_t)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        log_std = self.log_std(latent_pi)
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[Tuple[th.Tensor, th.Tensor], th.Tensor, th.Tensor]:
        assert self.use_sde, "use_sde True."
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        extra_pred = self.extra_pred(kwargs["latent_sde"])
        if self.extra_pred_dim == 7:
            extra_pred = th.cat(
                (F.normalize(extra_pred[:, :4], p=2, dim=-1), extra_pred[:, 4:]), dim=-1
            )
        elif self.extra_pred_dim == 9:
            extra_pred = th.cat(
                (
                    F.normalize(extra_pred[:, :3], p=2, dim=-1),
                    F.normalize(extra_pred[:, 3:6], p=2, dim=-1),
                    extra_pred[:, 6:],
                ),
                dim=-1,
            )
        else:
            raise NotImplementedError
        target_pred = self.target_pred(kwargs["latent_sde"])
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs), extra_pred, target_pred

    def features_forward(self, obs: th.Tensor):
        with th.no_grad():
            return self._extract_windowed_features(obs)


# ============================================================================
# 3. Critic -- same idea, per-frame extraction + real windowed LSTM
# ============================================================================
class CustomContinuousCritic(BaseModel):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        stack_keys: Optional[List[str]] = None,
        orig_observation_space: Optional[spaces.Dict] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        extra_pred_dim: int = 7,
        use_lstm: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        action_dim = get_action_dim(self.action_space)
        if orig_observation_space is None:
            orig_observation_space = observation_space
        if stack_keys is None:
            stack_keys = []
        self.stack_keys = stack_keys
        self.orig_observation_space = orig_observation_space
        if stack_keys:
            validate_stack_keys(orig_observation_space, stack_keys)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

        self.use_lstm = use_lstm
        # h_t = LSTM(z_{t-k+1..t}); Q_i = Q_i(h_t, a_t).
        # The window contains a_{t-1} (obs["prev_action"]); a_t is the Critic's `actions` input.
        self.temporal_encoder = TemporalEncoder(features_dim=features_dim, hidden_dim=features_dim)

        self.extra_pred_dim = extra_pred_dim
        self.extra_pred = nn.Linear(features_dim, extra_pred_dim)
        nn.init.xavier_uniform_(self.extra_pred.weight, gain=1)
        nn.init.constant_(self.extra_pred.bias, 0)

        self.target_pred = nn.Linear(features_dim, 4)
        nn.init.xavier_uniform_(self.target_pred.weight, gain=1)
        nn.init.constant_(self.target_pred.bias, 0)

    def _extract_sequence(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        """Stacked obs (batch, n_stack, ...) -> per-frame features z: [batch, n_stack, features_dim]."""
        if not self.stack_keys:
            with th.set_grad_enabled(not self.share_features_extractor):
                flat_obs = _prepare_unstacked_obs(obs, self.orig_observation_space)
                flat_obs = preprocess_obs(flat_obs, self.orig_observation_space, normalize_images=self.normalize_images)
                flat_obs = _ensure_action_aliases(flat_obs)
                feat = self.features_extractor(flat_obs)
                return feat.unsqueeze(1)
        with th.set_grad_enabled(not self.share_features_extractor):
            return extract_frame_sequence(
                self.features_extractor,
                obs,
                self.stack_keys,
                self.orig_observation_space,
                self.normalize_images,
            )

    def _extract_windowed_features(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        """h_t = LSTM(z_{t-k+1}, ..., z_t) if use_lstm else z_t"""
        if not self.stack_keys and not self.use_lstm:
            with th.set_grad_enabled(not self.share_features_extractor):
                flat_obs = _prepare_unstacked_obs(obs, self.orig_observation_space)
                flat_obs = preprocess_obs(flat_obs, self.orig_observation_space, normalize_images=self.normalize_images)
                flat_obs = _ensure_action_aliases(flat_obs)
                return self.features_extractor(flat_obs)
        seq = self._extract_sequence(obs)
        if self.use_lstm:
            return self.temporal_encoder(seq)
        return seq[:, -1, :]

    def forward(self, obs: th.Tensor, actions: th.Tensor):
        features = self._extract_windowed_features(obs)
        qvalue_input = th.cat([features, actions], dim=-1)

        extra_pred = None
        if self.extra_pred_dim:
            extra_pred = self.extra_pred(features)
            if self.extra_pred_dim == 7:
                extra_pred = th.cat(
                    (F.normalize(extra_pred[..., :4], p=2, dim=-1), extra_pred[..., 4:]), dim=-1
                )
            elif self.extra_pred_dim == 9:
                extra_pred = th.cat(
                    (
                        F.normalize(extra_pred[..., :3], p=2, dim=-1),
                        F.normalize(extra_pred[..., 3:6], p=2, dim=-1),
                        extra_pred[..., 6:],
                    ),
                    dim=-1,
                )
            else:
                raise NotImplementedError
        target_pred = self.target_pred(features)

        q_vals = tuple(q_net(qvalue_input) for q_net in self.q_networks)
        return q_vals, extra_pred, target_pred

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        with th.no_grad():
            features = self._extract_windowed_features(obs)
        qvalue_input = th.cat([features, actions], dim=-1)
        return self.q_networks[0](qvalue_input)

    def features_forward(self, obs: th.Tensor):
        with th.no_grad():
            return self._extract_windowed_features(obs)


# ============================================================================
# 4. Policy -- builds the features_extractor against the UNSTACKED space
# ============================================================================
class CustomSACPolicy(SACPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        orig_observation_space: Optional[spaces.Dict] = None,
        stack_keys: Optional[List[str]] = None,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        extra_pred_dim: int = 7,
        use_lstm: bool = False,
    ):
        if orig_observation_space is None:
            orig_observation_space = observation_space
        if stack_keys is None:
            stack_keys = []
        self.use_lstm = use_lstm
        self.orig_observation_space = orig_observation_space
        self.stack_keys = stack_keys
        self.extra_pred_dim = extra_pred_dim
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        kwargs = self.features_extractor_kwargs or {}
        return self.features_extractor_class(self.orig_observation_space, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self.actor_kwargs.copy()
        if features_extractor is None:
            features_extractor = self.make_features_extractor()
        actor_kwargs.update(
            dict(
                features_extractor=features_extractor,
                features_dim=features_extractor.features_dim,
                extra_pred_dim=self.extra_pred_dim,
                stack_keys=self.stack_keys,
                orig_observation_space=self.orig_observation_space,
                use_lstm=self.use_lstm,
                            )
        )
        return CustomActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomContinuousCritic:
        critic_kwargs = self.critic_kwargs.copy()
        if features_extractor is None:
            features_extractor = self.make_features_extractor()
        critic_kwargs.update(
            dict(
                features_extractor=features_extractor,
                features_dim=features_extractor.features_dim,
                extra_pred_dim=self.extra_pred_dim,
                stack_keys=self.stack_keys,
                orig_observation_space=self.orig_observation_space,
                use_lstm=self.use_lstm,
                            )
        )
        return CustomContinuousCritic(**critic_kwargs).to(self.device)

    def obs_to_tensor(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[th.Tensor, bool]:
        if isinstance(observation, dict):
            observation = dict(observation)
            if "prev_action" in observation and "prev_action" not in self.observation_space.spaces and "action" in self.observation_space.spaces:
                observation["action"] = observation["prev_action"]
            elif "action" in observation and "action" not in self.observation_space.spaces and "prev_action" in self.observation_space.spaces:
                observation["prev_action"] = observation["action"]
            observation = {k: v for k, v in observation.items() if k in self.observation_space.spaces}
        return super().obs_to_tensor(observation)


# ============================================================================
# 5. Loss helpers + CustomSAC
# ============================================================================
def qrot(q, v):
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)
    qvec = q[:, 1:]
    uv = th.cross(qvec, v, dim=1)
    uuv = th.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def transform_pred_grasp_pcs(grasp_pred, device="cuda"):
    grasp_shape = grasp_pred.shape
    assert grasp_shape[-1] == 7
    gripper_points = th.tensor(
        [
            [0, 0, -0.14],
            [0, 0, -0.07],
            [0.0425, 0, -0.07],
            [0.0425, 0, 0],
            [-0.0425, 0, -0.07],
            [-0.0425, 0, 0],
        ]
    ).to(device).float()
    adjust_gripper_points = gripper_points + th.tensor([0, 0, 0.02]).to(device)
    bs_gripper_points = th.unsqueeze(adjust_gripper_points, 0).repeat(grasp_shape[0], 1, 1)
    num_points = bs_gripper_points.shape[1]
    pred_grasps = th.unsqueeze(grasp_pred, 1).repeat(1, num_points, 1)
    pred_q = pred_grasps[:, :, :4]
    pred_t = pred_grasps[:, :, 4:]
    pred_gripper_points = qrot(pred_q, bs_gripper_points)
    pred_gripper_points += pred_t
    return pred_gripper_points


def goal_pred_loss(grasp_pred, grasp_goal, huber=False):
    goal_pcs = transform_pred_grasp_pcs(grasp_goal, device="cuda")
    pred_pcs = transform_pred_grasp_pcs(grasp_pred, device="cuda")
    return th.mean(th.abs(goal_pcs - pred_pcs).sum(-1))


def qmul(q0: th.Tensor, q1: th.Tensor) -> th.Tensor:
    w0, x0, y0, z0 = q0[..., 0], q0[..., 1], q0[..., 2], q0[..., 3]
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w = -x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1
    x = x0 * w1 + y0 * z1 - z0 * y1 + w0 * x1
    y = -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1
    z = x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1
    return th.stack((w, x, y, z), dim=-1)


def goal_pred_posquat_loss(grasp_pred, grasp_goal):
    trans_dist = th.norm((grasp_pred[:, 4:] - grasp_goal[:, 4:]), dim=1)
    rot_dist0 = 1 - th.clamp(th.abs(th.sum(grasp_pred[:, :4] * grasp_goal[:, :4], dim=-1)), min=0, max=1)
    grasp_goal_rotz180 = qmul(grasp_goal[:, :4], th.tensor([0.0, 0.0, 0.0, 1.0], device=grasp_goal.device))
    rot_dist1 = 1 - th.clamp(th.abs(th.sum(grasp_pred[:, :4] * grasp_goal_rotz180, dim=-1)), min=0, max=1)
    rotation_dist = th.minimum(rot_dist0, rot_dist1)
    return th.mean(trans_dist + rotation_dist)


def goal_pred_rotmat_loss(grasp_pred, grasp_goal):
    trans_dist = th.norm((grasp_pred[:, 6:] - grasp_goal[:, 6:]), dim=1)
    grasp_pred_rotz = th.cross(grasp_pred[:, :3], grasp_pred[:, 3:6])
    grasp_pred_mat = th.cat(
        (grasp_pred[:, :3].unsqueeze(2), grasp_pred[:, 3:6].unsqueeze(2), grasp_pred_rotz.unsqueeze(2)), dim=2
    )
    grasp_goal_rotz = th.cross(grasp_goal[:, :3], grasp_goal[:, 3:6])
    grasp_goal_mat = th.cat(
        (grasp_goal[:, :3].unsqueeze(2), grasp_goal[:, 3:6].unsqueeze(2), grasp_goal_rotz.unsqueeze(2)), dim=2
    )
    grasp_goal_mat_rotz180 = grasp_goal_mat.clone()
    grasp_goal_mat_rotz180 = grasp_goal_mat_rotz180 * th.tensor([-1, -1, 1], device=grasp_goal_mat_rotz180.device)
    rot_dist0 = 3 - th.diagonal(
        th.einsum("ijk, ikl -> ijl", grasp_pred_mat, grasp_goal_mat.transpose(1, 2)), dim1=-2, dim2=-1
    ).sum(dim=-1)
    rot_dist1 = 3 - th.diagonal(
        th.einsum("ijk, ikl -> ijl", grasp_pred_mat, grasp_goal_mat_rotz180.transpose(1, 2)), dim1=-2, dim2=-1
    ).sum(dim=-1)
    rotation_dist = th.minimum(rot_dist0, rot_dist1)
    return th.mean(trans_dist + rotation_dist)


def reward_target_loss(target_pred, target):
    return F.binary_cross_entropy_with_logits(target_pred, target)


class CustomSAC(SAC):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
        "CustomSACPolicy": CustomSACPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            _init_setup_model=_init_setup_model,
        )

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        if self.target_entropy == "auto":
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)

        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        actor_aux_losses, critic_aux_losses = [], []
        actor_target_losses, critic_target_losses = [], []

        aux_weight = 24.0
        target_weight = 24.0

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            close_grasp_pose_ee = replay_data.observations["close_grasp_pose_ee"]  # unstacked, (bs, 9)
            eval_target = replay_data.observations["eval_target"]  # unstacked, (bs, 4)

            if self.use_sde:
                self.actor.reset_noise()

            (actions_pi, log_prob), pred_pose_actor, pred_target_actor = self.actor.action_log_prob(replay_data.observations)
            actor_aux_loss = goal_pred_rotmat_loss(pred_pose_actor, close_grasp_pose_ee)
            actor_target_loss = reward_target_loss(pred_target_actor, eval_target)
            actor_aux_losses.append(actor_aux_loss.item())
            actor_target_losses.append(actor_target_loss.item())
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                (next_actions, next_log_prob), _, _ = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values, _, _ = self.critic_target(replay_data.next_observations, next_actions)
                next_q_values = th.cat(next_q_values, dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values, pred_pose_critic, pred_target_critic = self.critic(replay_data.observations, replay_data.actions)
            critic_aux_loss = goal_pred_rotmat_loss(pred_pose_critic, close_grasp_pose_ee)
            critic_target_loss = reward_target_loss(pred_target_critic, eval_target)
            critic_aux_losses.append(critic_aux_loss.item())
            critic_target_losses.append(critic_target_loss.item())

            critic_loss = (
                0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
                + critic_aux_loss * aux_weight
                + critic_target_loss * target_weight
            )
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            if getattr(self.critic, "use_lstm", False):
                th.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic.optimizer.step()

            q_values, _, _ = self.critic(replay_data.observations, actions_pi)
            q_values_pi = th.cat(q_values, dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean() + actor_aux_loss * aux_weight + actor_target_loss * target_weight
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            if getattr(self.actor, "use_lstm", False):
                th.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/actor_aux_loss", np.mean(actor_aux_losses))
        self.logger.record("train/critic_aux_loss", np.mean(critic_aux_losses))
        self.logger.record("train/actor_target_loss", np.mean(actor_target_losses))
        self.logger.record("train/critic_target_loss", np.mean(critic_target_losses))
        self.logger.record("train/aux_weight", aux_weight)
        self.logger.record("train/target_weight", target_weight)
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def set_parameters(
        self,
        load_path_or_dict: Union[str, Dict[str, Dict]],
        exact_match: bool = True,
        device: Union[th.device, str] = "auto",
    ) -> None:
        """Loads model parameters with automatic translation for legacy single-trunk/Experiment 6 checkpoints."""
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            from stable_baselines3.common.save_util import load_from_zip_file
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)

        if "policy" in params:
            remapped = {}
            for k, v in params["policy"].items():
                new_k = k
                if ".lstm." in new_k and ".temporal_encoder.lstm." not in new_k:
                    new_k = new_k.replace(".lstm.", ".temporal_encoder.lstm.")
                if ".state_map.weight" in new_k:
                    new_k = new_k.replace(".state_map.weight", ".state_map.0.weight")
                if ".state_map.bias" in new_k:
                    new_k = new_k.replace(".state_map.bias", ".state_map.0.bias")
                remapped[new_k] = v
            params["policy"] = remapped

            # If checkpoint has no LSTM / temporal_encoder weights, don't require exact match on temporal_encoder
            has_lstm = any(".lstm." in k or ".temporal_encoder." in k for k in remapped.keys())
            if not has_lstm:
                exact_match = False

        super().set_parameters(params, exact_match=exact_match, device=device)

    @classmethod
    def load(
        cls,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ):
        """Loads CustomSAC model with automatic observation space reconciliation for action/prev_action."""
        if custom_objects is None:
            custom_objects = {}
        else:
            custom_objects = dict(custom_objects)

        if env is not None and "action_space" not in custom_objects and hasattr(env, "action_space"):
            custom_objects["action_space"] = env.action_space

        from stable_baselines3.common.save_util import load_from_zip_file
        data, _, _ = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        if "action_space" not in data and env is not None and hasattr(env, "action_space"):
            custom_objects["action_space"] = env.action_space
        if "observation_space" not in data and env is not None and hasattr(env, "observation_space"):
            custom_objects["observation_space"] = env.observation_space

        if env is not None and isinstance(data.get("observation_space"), spaces.Dict):
            saved_keys = set(data["observation_space"].spaces.keys())
            target_key = "action" if "action" in saved_keys else ("prev_action" if "prev_action" in saved_keys else None)
            curr = env
            while curr is not None:
                if hasattr(curr, "align_action_key") and target_key:
                    curr.align_action_key(target_key)
                elif hasattr(curr, "observation_space") and isinstance(curr.observation_space, spaces.Dict):
                    curr_spaces = dict(curr.observation_space.spaces)
                    if target_key == "action" and "prev_action" in curr_spaces:
                        curr_spaces["action"] = curr_spaces.get("action", curr_spaces["prev_action"])
                        curr_spaces.pop("prev_action", None)
                    elif target_key == "prev_action" and "action" in curr_spaces:
                        curr_spaces["prev_action"] = curr_spaces.get("prev_action", curr_spaces["action"])
                        curr_spaces.pop("action", None)
                    curr.observation_space = spaces.Dict({k: curr_spaces[k] for k in saved_keys if k in curr_spaces})
                curr = getattr(curr, "env", None)

        return super().load(
            path,
            env=env,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
            force_reset=force_reset,
            **kwargs,
        )

