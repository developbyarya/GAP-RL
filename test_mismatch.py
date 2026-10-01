import gym
from gym import spaces
import numpy as np

AUX_OBS_KEYS = ("close_grasp_pose_ee", "eval_target")

def default_stack_keys(observation_space: spaces.Dict):
    return [k for k in observation_space.spaces.keys() if k not in AUX_OBS_KEYS]

class FrameStackObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_stack, stack_keys):
        super().__init__(env)
        self.n_stack = n_stack
        self.stack_keys = stack_keys
        
        new_spaces = {}
        for key, space in env.observation_space.spaces.items():
            if key in stack_keys:
                low = np.repeat(space.low[None], n_stack, axis=0)
                high = np.repeat(space.high[None], n_stack, axis=0)
                new_spaces[key] = spaces.Box(low=low, high=high, dtype=space.dtype)
            else:
                new_spaces[key] = space
        self.observation_space = spaces.Dict(new_spaces)
        
    def observation(self, obs): return obs

class DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict({
            "action": spaces.Box(-np.inf, np.inf, (7,), np.float32),
            "close_grasp_pose_ee": spaces.Box(-np.inf, np.inf, (9,), np.float32),
            "eval_target": spaces.Box(-np.inf, np.inf, (4,), np.float32),
            "grasp_exist": spaces.Box(-np.inf, np.inf, (5,), np.float32),
            "grasps_scores": spaces.Box(-np.inf, np.inf, (32,), np.float32),
        })
    def reset(self): return {}
    def step(self, a): return {}, 0, False, {}

class FrameStackWrapper(gym.Wrapper): # mimic VecEnvWrapper
    def __init__(self, env, n_stack, stack_keys):
        super().__init__(env)
        new_spaces = {}
        for key, space in env.observation_space.spaces.items():
            if key in stack_keys:
                low = np.repeat(space.low[None], n_stack, axis=0)
                high = np.repeat(space.high[None], n_stack, axis=0)
                new_spaces[key] = spaces.Box(low=low, high=high, dtype=space.dtype)
            else:
                new_spaces[key] = space
        self.observation_space = spaces.Dict(new_spaces)

env1 = DummyEnv()
keys1 = default_stack_keys(env1.observation_space)
env1_stacked = FrameStackObsWrapper(env1, 4, keys1)

env2 = DummyEnv()
keys2 = default_stack_keys(env2.observation_space)
env2_stacked = FrameStackWrapper(env2, 4, keys2)

print(env1_stacked.observation_space == env2_stacked.observation_space)
