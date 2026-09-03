import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "gap_rl", "algorithms", "scripts")))

import gym
import gap_rl.envs
from gap_rl.utils.wrappers.common import NormalizeBoxActionWrapper
from custom_sac import FrameStackObsWrapper, FrameStackWrapper, default_stack_keys
from stable_baselines3.common.vec_env import DummyVecEnv

# 1. Eval way
env = gym.make("PickSingleEGAD-v0", obs_mode="state_egopoints_rt")
env = NormalizeBoxActionWrapper(env)
stack_keys = default_stack_keys(env.observation_space)
eval_env = FrameStackObsWrapper(env, n_stack=4, stack_keys=stack_keys)
eval_space = eval_env.observation_space

# 2. Train way
def make_env():
    e = gym.make("PickSingleEGAD-v0", obs_mode="state_egopoints_rt")
    e = NormalizeBoxActionWrapper(e)
    return e

venv = DummyVecEnv([make_env])
train_venv = FrameStackWrapper(venv, n_stack=4, stack_keys=stack_keys)
train_space = train_venv.observation_space

print(eval_space == train_space)
if eval_space != train_space:
    print("Mismatched keys:")
    for k in eval_space.spaces.keys():
        if eval_space.spaces[k] != train_space.spaces[k]:
            print(k)
            print("Eval:", eval_space.spaces[k])
            print("Train:", train_space.spaces[k])
