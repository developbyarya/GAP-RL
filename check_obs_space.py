import zipfile
import json
import cloudpickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "gap_rl", "algorithms", "scripts")))

model_path = "runs/reward_ablation_20260901_110135_iew0.3/rl_model_2000000_steps.zip"

with zipfile.ZipFile(model_path, "r") as archive:
    data_json = archive.read("data").decode("utf-8")
    data = json.loads(data_json)
    obs_space_b64 = data["observation_space"]
    
import base64
obs_space = cloudpickle.loads(base64.b64decode(obs_space_b64))

print("Expected by model:")
for k, v in obs_space.spaces.items():
    print(f"  {k}: {v.shape}")
