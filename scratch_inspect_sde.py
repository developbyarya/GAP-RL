import stable_baselines3
from stable_baselines3.sac.policies import Actor
import gym
from gym import spaces
import torch.nn as nn
from stable_baselines3.common.torch_layers import NatureCNN

# We will just print the actor
obs_space = spaces.Box(-1, 1, (4,))
act_space = spaces.Box(-1, 1, (2,))

actor = Actor(obs_space, act_space, [256, 256], nn.Identity(), 4, use_sde=True)
print("use_sde=True mu:", type(actor.mu))
print(actor.mu)

actor2 = Actor(obs_space, act_space, [256, 256], nn.Identity(), 4, use_sde=False)
print("use_sde=False mu:", type(actor2.mu))
print(actor2.mu)

