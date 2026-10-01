import re

with open("gap_rl/envs/pick_single.py", "r") as f:
    content = f.read()

# 1. Add DEFAULT_LOCALGRASP_JSON and __init__ to PickSingleGraspnetEnv
init_code = """    DEFAULT_LOCALGRASP_JSON = "info_localgrasp_v0.json"

    def __init__(self, **kwargs):
        from pathlib import Path
        import json
        from collections import OrderedDict
        asset_root = Path(self.DEFAULT_ASSET_ROOT.format(ASSET_DIR="gap_rl/data"))
        # We need to manually load the lg_grasps from the JSON.
        lg_json = asset_root / self.DEFAULT_LOCALGRASP_JSON
        if not lg_json.exists():
            raise FileNotFoundError(f"{lg_json} is not found. Please run gen_LoG_grasps.py first!")
        
        with open(lg_json, 'r') as f:
            lg_db = json.load(f)
            
        all_lg_grasps = OrderedDict()
        for model_id in kwargs.get("model_ids", []):
            all_lg_grasps[model_id] = lg_db[model_id]["grasp"]
        self.all_lg_grasps = all_lg_grasps
        super().__init__(**kwargs)"""

content = re.sub(
    r'(class PickSingleGraspnetEnv\(PickSingleEnv\):\n    DEFAULT_ASSET_ROOT = "[^"]+"\n    DEFAULT_MODEL_JSON = "[^"]+"\n    DEFAULT_GRASP_JSON = "[^"]+")',
    r'\1\n' + init_code,
    content
)

# 2. Add the lg_grasps_dict loading logic to PickSingleGraspnetEnv._load_model
# Find PickSingleGraspnetEnv._load_model
start_idx = content.find("def _load_model(self):", content.find("class PickSingleGraspnetEnv"))
end_idx = content.find("def _load_actors(self):", start_idx)

load_logic = """
        if self.obs_mode in ["state_egopoints", "state_grasp9d", "state_grasp_obj_points"]:
            self.lg_grasps_dict = self.all_lg_grasps[self.model_id]
            grasp_views = len(self.lg_grasps_dict)
            if self.grasp_select_mode in ["random", "angle_filter"]:
                lg_grasps_poses = []
                lg_grasps_scores = []
                for grasp_view_id in range(grasp_views):
                    if self.lg_grasps_dict[grasp_view_id] is not None:
                        transformations = self.lg_grasps_dict[grasp_view_id]['transformations']
                        lg_grasps_poses.extend(transformations)
                        scores = self.lg_grasps_dict[grasp_view_id]['scores']
                        lg_grasps_scores.extend(scores)
                if len(lg_grasps_poses) > 0:
                    lg_grasps_poses = np.array(lg_grasps_poses)
                    grasp_mats = np.repeat(np.eye(4)[None], lg_grasps_poses.shape[0], 0)
                    grasp_mats[:, :3, 3] = lg_grasps_poses[:, :3]
                    from scipy.spatial.transform import Rotation
                    grasp_mats[:, :3, :3] = Rotation.from_quat(lg_grasps_poses[:, 3:]).as_matrix()
                    self.lg_grasps_mat = grasp_mats
                    self.lg_grasps_score = np.array(lg_grasps_scores)
                else:
                    self.lg_grasps_mat = np.zeros((0, 4, 4))
                    self.lg_grasps_score = np.zeros(0)
"""

sub_content = content[start_idx:end_idx]
sub_content = sub_content.replace(
    "self.obj_aabb_halfsize = self.obj_bbdx.extents / 2\n",
    "self.obj_aabb_halfsize = self.obj_bbdx.extents / 2\n" + load_logic + "\n"
)

content = content[:start_idx] + sub_content + content[end_idx:]

with open("gap_rl/envs/pick_single.py", "w") as f:
    f.write(content)
