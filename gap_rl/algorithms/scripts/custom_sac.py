"""
Frame-stacked version of CustomSAC actor/critic.

Uses a rolling observation window (FrameStackWrapper / FrameStackObsWrapper)
so the LSTM receives a real (batch, n_stack, features_dim) sequence instead of
a single zero-state timestep from i.i.d. ReplayBuffer samples.
"""

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
from gap_rl.algorithms.grasp_tracking import GraspTracking

SelfSAC = TypeVar("SelfSAC", bound="SAC")

# Aux supervision targets: current-step only, never stacked into history.
AUX_OBS_KEYS = ("close_grasp_pose_ee", "eval_target")


def default_stack_keys(observation_space: spaces.Dict) -> List[str]:
    """Stack every Dict key except aux supervision targets."""
    return [k for k in observation_space.spaces.keys() if k not in AUX_OBS_KEYS]

def reconstruct_gripper_pts_diff(grasps_posrot_ee, origin_gripper_pts):
    B, ng, _ = grasps_posrot_ee.shape
    _, k, _ = origin_gripper_pts.shape
    R_x = grasps_posrot_ee[:, :, 3:6]
    R_y = grasps_posrot_ee[:, :, 6:9]
    R_z = th.cross(R_x, R_y, dim=-1)
    R = th.stack([R_x, R_y, R_z], dim=-1) # (B, ng, 3, 3)
    T = grasps_posrot_ee[:, :, :3]
    diff = th.einsum("bki, bnji -> bnkj", origin_gripper_pts, R) + T.unsqueeze(2)
    return diff


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

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()  # gym==0.21 4-tuple
        stacked = []
        for i in range(self.n_envs):
            single = {k: v[i] for k, v in obs.items()}
            if dones[i]:
                # VecEnv auto-resets on done: this `obs` is already the fresh
                # reset observation, so re-seed the window rather than push
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
        self.stack_keys = stack_keys
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

    def _init_history(self, obs: Dict[str, np.ndarray]):
        for k in self.stack_keys:
            self._history[k].clear()
            for _ in range(self.n_stack):
                self._history[k].append(obs[k])

    def _push(self, obs: Dict[str, np.ndarray]):
        for k in self.stack_keys:
            self._history[k].append(obs[k])

    def _build_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = {}
        for k, v in obs.items():
            out[k] = np.stack(self._history[k], axis=0) if k in self.stack_keys else v
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
        stack_keys: List[str],
        orig_observation_space: spaces.Dict,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        extra_pred_dim: int = 7,
        use_grasp_tracking: bool = False,
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
        self.stack_keys = stack_keys
        self.orig_observation_space = orig_observation_space

        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
        self.lstm = nn.LSTM(features_dim, features_dim, batch_first=True)
        self.extra_pred = nn.Linear(last_layer_dim, extra_pred_dim)
        nn.init.xavier_uniform_(self.extra_pred.weight, gain=1)
        nn.init.constant_(self.extra_pred.bias, 0)
        self.extra_pred_dim = extra_pred_dim

        self.target_pred = nn.Linear(last_layer_dim, 4)
        nn.init.xavier_uniform_(self.target_pred.weight, gain=1)
        nn.init.constant_(self.target_pred.bias, 0)
        
        self.use_grasp_tracking = use_grasp_tracking
        if self.use_grasp_tracking:
            self.grasp_tracker = GraspTracking(pose_dim=9, hidden_size=16)

    def _extract_windowed_features(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        """
        obs[key] for key in stack_keys has shape (batch, n_stack, *orig_shape).
        Slice each timestep, run the per-frame features_extractor, LSTM over the
        real sequence.
        """
        n_stack = obs[self.stack_keys[0]].shape[1]
        frame_feats = []
        
        tracker_hidden = None
        prev_target = None
        
        for t in range(n_stack):
            obs_t = {k: (v[:, t] if k in self.stack_keys else v) for k, v in obs.items()}
            
            if self.use_grasp_tracking and "grasps_posrot_ee" in obs_t:
                candidates = obs_t["grasps_posrot_ee"]
                scores = obs_t["grasps_scores"]
                
                tracked_pose, conf, runner_ups, tracker_hidden = self.grasp_tracker(
                    candidates, scores, prev_target, tracker_hidden
                )
                prev_target = tracked_pose
                
                new_grasps_posrot = th.cat([tracked_pose.unsqueeze(1), runner_ups], dim=1)
                obs_t["grasps_posrot_ee"] = new_grasps_posrot
                obs_t["grasp_tracking_conf"] = conf
                
                if "gripper_pts_diff" in obs_t and "origin_gripper_pts" in obs_t:
                    obs_t["gripper_pts_diff"] = reconstruct_gripper_pts_diff(
                        new_grasps_posrot, obs_t["origin_gripper_pts"]
                    )
                
            obs_t = preprocess_obs(obs_t, self.orig_observation_space, normalize_images=self.normalize_images)
            frame_feats.append(self.features_extractor(obs_t))
        seq = th.stack(frame_feats, dim=1)  # (batch, n_stack, features_dim)
        lstm_out, _ = self.lstm(seq)
        return lstm_out[:, -1, :]

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        features = self._extract_windowed_features(obs)
        latent_pi = self.latent_pi(features)
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
        stack_keys: List[str],
        orig_observation_space: spaces.Dict,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        extra_pred_dim: int = 7,
        use_grasp_tracking: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        action_dim = get_action_dim(self.action_space)
        self.stack_keys = stack_keys
        self.orig_observation_space = orig_observation_space

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

        self.lstm = nn.LSTM(features_dim, features_dim, batch_first=True)

        self.extra_pred_dim = extra_pred_dim
        self.extra_pred = nn.Linear(features_dim, extra_pred_dim)
        nn.init.xavier_uniform_(self.extra_pred.weight, gain=1)
        nn.init.constant_(self.extra_pred.bias, 0)

        self.target_pred = nn.Linear(features_dim, 4)
        nn.init.xavier_uniform_(self.target_pred.weight, gain=1)
        nn.init.constant_(self.target_pred.bias, 0)
        
        self.use_grasp_tracking = use_grasp_tracking
        if self.use_grasp_tracking:
            self.grasp_tracker = GraspTracking(pose_dim=9, hidden_size=16)

    def _extract_windowed_features(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        n_stack = obs[self.stack_keys[0]].shape[1]
        frame_feats = []
        
        tracker_hidden = None
        prev_target = None
        
        with th.set_grad_enabled(not self.share_features_extractor):
            for t in range(n_stack):
                obs_t = {k: (v[:, t] if k in self.stack_keys else v) for k, v in obs.items()}
                
                if self.use_grasp_tracking and "grasps_posrot_ee" in obs_t:
                    candidates = obs_t["grasps_posrot_ee"]
                    scores = obs_t["grasps_scores"]
                    
                    tracked_pose, conf, runner_ups, tracker_hidden = self.grasp_tracker(
                        candidates, scores, prev_target, tracker_hidden
                    )
                    prev_target = tracked_pose
                    
                    obs_t["grasps_posrot_ee"] = th.cat([tracked_pose.unsqueeze(1), runner_ups], dim=1)
                    obs_t["grasp_tracking_conf"] = conf
                    
                obs_t = preprocess_obs(obs_t, self.orig_observation_space, normalize_images=self.normalize_images)
                frame_feats.append(self.features_extractor(obs_t))
        seq = th.stack(frame_feats, dim=1)
        lstm_out, _ = self.lstm(seq)
        return lstm_out[:, -1, :]

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
        orig_observation_space: spaces.Dict,
        stack_keys: List[str],
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
        use_grasp_tracking: bool = False,
    ):
        self.orig_observation_space = orig_observation_space
        self.stack_keys = stack_keys
        self.extra_pred_dim = extra_pred_dim
        self.use_grasp_tracking = use_grasp_tracking
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
        if self.use_grasp_tracking:
            kwargs["use_grasp_tracking"] = True
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
                use_grasp_tracking=self.use_grasp_tracking,
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
                use_grasp_tracking=self.use_grasp_tracking,
            )
        )
        return CustomContinuousCritic(**critic_kwargs).to(self.device)


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
        mask_early_episode_loss: bool = False,
        n_stack: int = 4,
    ):
        self.mask_early_episode_loss = mask_early_episode_loss
        self.n_stack = n_stack
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

        aux_weight = 100 * self.gamma ** (self.num_timesteps // 20000)
        target_weight = 100 * self.gamma ** (self.num_timesteps // 20000)

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            close_grasp_pose_ee = replay_data.observations["close_grasp_pose_ee"]  # unstacked, (bs, 9)
            eval_target = replay_data.observations["eval_target"]  # unstacked, (bs, 4)
            
            # Mask early episode transitions
            loss_mask = th.ones(batch_size, dtype=th.bool, device=self.device)
            if getattr(self, "mask_early_episode_loss", False) and getattr(self, "n_stack", 1) > 1:
                key = self.policy.stack_keys[0]
                frame0 = replay_data.observations[key][:, 0].view(batch_size, -1)
                frame1 = replay_data.observations[key][:, 1].view(batch_size, -1)
                diff = th.abs(frame0 - frame1).sum(dim=1)
                loss_mask = diff > 1e-6
                
                # Ensure we have at least one valid sample to avoid NaNs
                if not loss_mask.any():
                    loss_mask[0] = True

            if self.use_sde:
                self.actor.reset_noise()

            (actions_pi, log_prob), pred_pose_actor, pred_target_actor = self.actor.action_log_prob(replay_data.observations)
            
            # Apply mask to actor aux loss
            actor_aux_loss = goal_pred_rotmat_loss(pred_pose_actor, close_grasp_pose_ee)
            if self.mask_early_episode_loss: actor_aux_loss = (actor_aux_loss * loss_mask).sum() / loss_mask.sum()
            
            actor_target_loss = reward_target_loss(pred_target_actor, eval_target)
            if self.mask_early_episode_loss: actor_target_loss = (actor_target_loss * loss_mask).sum() / loss_mask.sum()
            
            actor_aux_losses.append(actor_aux_loss.item())
            actor_target_losses.append(actor_target_loss.item())
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                # Mask entropy loss? Entropy loss depends only on current policy log_prob, we should probably mask it too
                if self.mask_early_episode_loss:
                    ent_coef_loss = -(self.log_ent_coef * (log_prob[loss_mask] + self.target_entropy).detach()).mean()
                else:
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
            if self.mask_early_episode_loss: critic_aux_loss = (critic_aux_loss * loss_mask).sum() / loss_mask.sum()
            
            critic_target_loss = reward_target_loss(pred_target_critic, eval_target)
            if self.mask_early_episode_loss: critic_target_loss = (critic_target_loss * loss_mask).sum() / loss_mask.sum()
            
            critic_aux_losses.append(critic_aux_loss.item())
            critic_target_losses.append(critic_target_loss.item())

            # Mask critic TD loss
            if self.mask_early_episode_loss:
                td_losses = [F.mse_loss(current_q, target_q_values, reduction='none') for current_q in current_q_values]
                td_loss = sum((td[loss_mask]).mean() for td in td_losses) * 0.5
            else:
                td_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)

            critic_loss = (
                td_loss
                + critic_aux_loss * aux_weight
                + critic_target_loss * target_weight
            )
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values, _, _ = self.critic(replay_data.observations, actions_pi)
            q_values_pi = th.cat(q_values, dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            
            if self.mask_early_episode_loss:
                actor_td_loss = (ent_coef * log_prob - min_qf_pi)[loss_mask].mean()
            else:
                actor_td_loss = (ent_coef * log_prob - min_qf_pi).mean()
                
            actor_loss = actor_td_loss + actor_aux_loss * aux_weight + actor_target_loss * target_weight
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
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
