import stable_baselines3
import inspect
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.sac import SAC

print("=== SACPolicy._build ===")
print(inspect.getsource(SACPolicy._build))

print("=== SAC._setup_model ===")
print(inspect.getsource(SAC._setup_model))

print("=== SAC._get_torch_save_params ===")
try:
    print(inspect.getsource(SAC._get_torch_save_params))
except AttributeError:
    # It might be in BaseAlgorithm
    from stable_baselines3.common.base_class import BaseAlgorithm
    print(inspect.getsource(BaseAlgorithm._get_torch_save_params))

