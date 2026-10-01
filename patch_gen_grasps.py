import re

with open("gap_rl/examples/gen_LoG_grasps_graspnet.py", "r") as f:
    content = f.read()

# Add missing imports
content = content.replace(
    "from gap_rl.utils.geometry import homo_transfer, transform_points, sample_grasp_points_ee",
    "from gap_rl.utils.geometry import homo_transfer, transform_points, sample_grasp_points_ee, pointcloud_filter, pc_bbdx_filter"
)

# Replace the data_cam block
old_block = """            data_cam_pose = look_at(eye=pos_it, target=sample_poscenter, up=[0, 0, 1])
            env.unwrapped._cameras["data_cam"].camera.set_pose(data_cam_pose)
            
            scene_pc_ee, obj_pc_ee = env.unwrapped.get_objpoints_rt()
            
            if obj_pc_ee.shape[0] < 64:"""

new_block = """            data_cam_pose = look_at(eye=pos_it, target=sample_poscenter, up=[0, 0, 1])
            env.unwrapped._cameras["data_cam"].camera.set_pose(data_cam_pose)
            
            cam = env.unwrapped._cameras["data_cam"]
            env.unwrapped.update_render()
            cam.take_picture()
            trans_cam2world = cam.camera.get_model_matrix()
            cam_pc = cam.get_camera_pcd(rgb=False, visual_seg=False, actor_seg=False)['xyz']
            trans_world2ee = env.unwrapped.trans_cam2ee @ np.linalg.inv(trans_cam2world)
            scene_pc = transform_points(trans_cam2world, cam_pc)
            
            ground_ws = ([-0.5, 0.5], [-0.5, 0.5], [-0.0001, 0.5])
            scene_pc, mask = pointcloud_filter(scene_pc, ground_ws)
            scenepcee = transform_points(trans_world2ee, scene_pc)
            
            obj_bbdx_v = env.unwrapped.obj_bbdx.vertices
            trans_world2obj = env.unwrapped.obj_pose.inv().to_transformation_matrix()
            scene_pc_obj = transform_points(trans_world2obj, scene_pc)
            _, mask = pc_bbdx_filter(scene_pc_obj, obj_bbdx_v)
            obj_pc_ee = scenepcee[mask]
            scene_pc_ee = scenepcee
            
            if obj_pc_ee.shape[0] < 64:"""

content = content.replace(old_block, new_block)

with open("gap_rl/examples/gen_LoG_grasps_graspnet.py", "w") as f:
    f.write(content)
