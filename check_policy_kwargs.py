import zipfile
import json
import cloudpickle
import base64
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "gap_rl", "algorithms", "scripts")))

model_path = "runs/reward_ablation_20260901_110135_iew0.3/rl_model_2000000_steps.zip"

with zipfile.ZipFile(model_path, "r") as archive:
    data_json = archive.read("data").decode("utf-8")
    data = json.loads(data_json)
    
if "policy_kwargs" in data:
    pk_str = data["policy_kwargs"]
    if isinstance(pk_str, dict) and "_type" in pk_str and pk_str["_type"] == "cloudpickle":
        pk = cloudpickle.loads(base64.b64decode(pk_str["data"]))
        print("policy_kwargs found! Keys:", pk.keys())
    elif isinstance(pk_str, str):
        print("policy_kwargs is a string?!", pk_str[:100])
    else:
        print("policy_kwargs:", pk_str.keys() if isinstance(pk_str, dict) else pk_str)
else:
    print("NO policy_kwargs in data!!")
