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

SelfSAC = TypeVar("SelfSAC", bound="SAC")

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
        orig_observation_space: spaces.Dict,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        extra_pred_dim: int = 7,
        use_attn_lstm: bool = False,
        d_attn: int = 384,
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
        self.orig_observation_space = orig_observation_space

        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
        self.use_attn_lstm = use_attn_lstm
        self.token_attn = TokenSelfAttention(d_attn=d_attn, use_attn_lstm=use_attn_lstm)
        lstm_input_dim = 5 * d_attn if use_attn_lstm else features_dim
        self.lstm = nn.LSTM(lstm_input_dim, features_dim, batch_first=True)
        self.extra_pred = nn.Linear(last_layer_dim, extra_pred_dim)
        nn.init.xavier_uniform_(self.extra_pred.weight, gain=1)
        nn.init.constant_(self.extra_pred.bias, 0)
        self.extra_pred_dim = extra_pred_dim

        self.target_pred = nn.Linear(last_layer_dim, 4)
        nn.init.xavier_uniform_(self.target_pred.weight, gain=1)
        nn.init.constant_(self.target_pred.bias, 0)

    def _extract_windowed_features(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        obs_pre = preprocess_obs(obs, self.orig_observation_space, normalize_images=self.normalize_images)
        feat = self.features_extractor(obs_pre)
        feat = self.token_attn(obs_pre, feat)
        seq = feat.unsqueeze(1)  # (batch, 1, features_dim)
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
        orig_observation_space: spaces.Dict,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        extra_pred_dim: int = 7,
        use_attn_lstm: bool = False,
        d_attn: int = 384,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        action_dim = get_action_dim(self.action_space)
        self.orig_observation_space = orig_observation_space

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

        self.use_attn_lstm = use_attn_lstm
        self.token_attn = TokenSelfAttention(d_attn=d_attn, use_attn_lstm=use_attn_lstm)
        lstm_input_dim = 5 * d_attn if use_attn_lstm else features_dim
        self.lstm = nn.LSTM(lstm_input_dim, features_dim, batch_first=True)

        self.extra_pred_dim = extra_pred_dim
        self.extra_pred = nn.Linear(features_dim, extra_pred_dim)
        nn.init.xavier_uniform_(self.extra_pred.weight, gain=1)
        nn.init.constant_(self.extra_pred.bias, 0)

        self.target_pred = nn.Linear(features_dim, 4)
        nn.init.xavier_uniform_(self.target_pred.weight, gain=1)
        nn.init.constant_(self.target_pred.bias, 0)

    def _extract_windowed_features(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        with th.set_grad_enabled(not self.share_features_extractor):
            obs_pre = preprocess_obs(obs, self.orig_observation_space, normalize_images=self.normalize_images)
            feat = self.features_extractor(obs_pre)
            feat = self.token_attn(obs_pre, feat)
        seq = feat.unsqueeze(1)
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
        use_attn_lstm: bool = False,
        d_attn: int = 384,
    ):
        self.use_attn_lstm = use_attn_lstm
        self.d_attn = d_attn
        self.orig_observation_space = orig_observation_space
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
                orig_observation_space=self.orig_observation_space,
                use_attn_lstm=self.use_attn_lstm,
                d_attn=self.d_attn,
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
                orig_observation_space=self.orig_observation_space,
                use_attn_lstm=self.use_attn_lstm,
                d_attn=self.d_attn,
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

        aux_weight = 100 * self.gamma ** (self.num_timesteps // 20000)
        target_weight = 100 * self.gamma ** (self.num_timesteps // 20000)

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
            self.critic.optimizer.step()

            q_values, _, _ = self.critic(replay_data.observations, actions_pi)
            q_values_pi = th.cat(q_values, dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean() + actor_aux_loss * aux_weight + actor_target_loss * target_weight
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
