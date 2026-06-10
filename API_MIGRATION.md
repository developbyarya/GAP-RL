Good, I now have enough information from the docs to produce a comprehensive mapping table. Let me compile everything, including notes on the architectural shift (Engine+Scene → Scene-only, Actor → Entity+Component, etc.).

Here's the complete API mapping table based on the SAPIEN 3 documentation:

---

## SAPIEN 2 → SAPIEN 3 API Mapping

| SAPIEN 2 API | SAPIEN 3 Equivalent | Notes |
|---|---|---|
| `sapien.Engine()` | *(removed)* | SAPIEN 3 underwent a major API & infrastructure overhaul. The `Engine` singleton is gone. Scene creation is now direct: `scene = sapien.Scene()`. Physics and render systems are managed internally or explicitly via `sapien.physx.PhysxCpuSystem()` / `sapien.render.RenderSystem()`. |
| `sapien.SapienRenderer` | `sapien.render.RenderSystem()` / `sapien.render.set_global_config(...)` | `SapienRenderer` still exists as a compatibility alias in some builds, but the canonical S3 pattern skips it entirely. To explicitly disable on-screen rendering, call `sapien.render.set_global_config(offscreen_only=True)` at the beginning of the program. No `engine.set_renderer(renderer)` call needed. |
| `sapien.render_config` | `sapien.render.set_global_config(...)` | Use `from sapien.render import set_global_config` then `set_global_config(max_num_materials=..., max_num_textures=...)`. RT settings moved to `sapien.render.set_ray_tracing_*` helpers. |
| `sapien.Pose` | `sapien.Pose` | Unchanged. Still `sapien.Pose(p=[x,y,z], q=[w,x,y,z])`. |
| `sapien.SceneConfig` | `sapien.physx.PhysxSceneConfig` | `scene_config = sapien.physx.PhysxSceneConfig()` — modify gravity, friction, restitution here, then call `sapien.physx.set_scene_config(scene_config)` before creating the scene. The S2 pattern `engine.create_scene(scene_config)` is gone; config must be set before `sapien.Scene()` is constructed. |
| `scene.create_drive(...)` | `joint.set_drive_properties(...)` on a `PhysxArticulationJointComponent` | A built-in PD controller is supported by revolute and prismatic joints — use `set_drive_properties` to specify stiffness and damping. Cross-body drives (between two arbitrary actors) no longer use a separate `Drive` object; use `PhysxDriveComponent` attached to an entity instead. |
| `scene.add_camera(...)` / `scene.add_mounted_camera(...)` | `scene.add_camera(...)` / `scene.add_mounted_camera(...)` (signature change) | Both methods still exist. `scene.add_camera` creates an Entity with a `RenderCameraComponent` and `camera.entity.set_pose(sapien.Pose(...))` sets its world pose. `scene.add_mounted_camera` now takes a `pose` arg that sets the **local** pose relative to the mount actor. ⚠️ Use `camera.entity.set_pose(...)` not `camera.set_pose(...)` to avoid confusion between local and world pose. |
| `scene.create_urdf_loader()` / `URDFLoader.load(...)` | `scene.create_urdf_loader()` / `loader.load(urdf_path)` | Interface is largely the same: `loader = scene.create_urdf_loader(); loader.fix_root_link = True; robot = loader.load("path/to.urdf")`. ⚠️ Returns a `PhysxArticulation` (not `Articulation`). `robot.set_root_pose(...)` replaces the old `articulation.set_root_pose(...)`. |
| `articulation.create_pinocchio_model()` | `mplib.planner.PinocchioModel(urdf_path)` | Pinocchio is now accessed through `mplib` (the separate motion planning library): `from mplib.kinematics.pinocchio import PinocchioModel; model = PinocchioModel(urdf_filename=...)`. ⚠️ No longer a method on the articulation object — you must pass the URDF path directly. IK result tuple structure (`result, success, error`) is unchanged. |
| `sapien.utils.Viewer` | `scene.create_viewer()` | In S3, `viewer = scene.create_viewer()` is the idiomatic way. `from sapien.utils import Viewer` still works but requires manual `viewer.set_scene(scene)` binding. `viewer.window.set_camera_parameters(...)` and `set_camera_xyz/rpy` are unchanged. |
| `sapien.RenderServer` | **Removed / no direct equivalent** | `RenderServer` was a SAPIEN 2.2 feature for multi-process rendering. In S3 the render system is unified. For parallel / headless rendering at scale, the recommended path is [ManiSkill's GPU-parallel pipeline](https://github.com/haosulab/ManiSkill) built on SAPIEN 3. |
| `_renderer._internal_context.create_line_set(...)` | `sapien.render.RenderCameraComponent` + custom line visual entity | The internal Vulkan context API is not exposed in S3. Line/point cloud rendering is done by building a visual entity with a `RenderBodyComponent` using a line mesh, or using `scene.create_actor_builder()` with a custom visual. No direct 1:1 drop-in. |
| `sapien.sensor.StereoDepthSensor` | `sapien.sensor.StereoDepthSensor` (partially ported) | The class still exists at `from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig`. However the docs warn this section has not been fully ported to S3 and some functions may be renamed. ⚠️ The old S2 init pattern (`sim = sapien.Engine(); renderer = sapien.SapienRenderer()`) in the existing doc is stale — replace with the S3 scene setup. The `sensor.take_picture()` / `sensor.compute_depth()` / `sensor.get_depth()` / `sensor.get_pointcloud()` call signatures appear unchanged. |
| `ActorBuilder.add_multiple_collisions_from_file(...)` | `ActorBuilder.add_multiple_convex_collisions_from_file(filename=..., scale=...)` | The S3 method is `builder.add_multiple_convex_collisions_from_file(filename=str(path), scale=scale)`. ⚠️ Name change: `multiple_collisions` → `multiple_convex_collisions`. Also note the old `color`/`material` split in visual shapes is unified to a single `material` param in S3 for `add_*_visual` methods. |

---

### Key Architectural Shifts to Keep in Mind

Two changes affect almost everything in the codebase:

**1. Actor → Entity + Component**

`Actor` from SAPIEN 2 is now `Entity` in SAPIEN 3. The functionalities of an Actor now become *components* attached to an entity. So anywhere GAP-RL does `actor.get_pose()`, `actor.set_pose()`, etc., those still work through the entity interface (`entity.get_pose()`) or via the attached `PhysxRigidDynamicComponent`. Physics properties like damping are now accessed via `actor.find_component_by_type(sapien.physx.PhysxRigidBodyComponent)`.

**2. Engine+Scene creation → Scene only**

S2 pattern:
```python
engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
scene = engine.create_scene(scene_config)
```
S3 pattern:
```python
sapien.physx.set_scene_config(scene_config)  # optional, before scene creation
scene = sapien.Scene()
```

These two shifts will cause the most cascading changes through the GAP-RL environment wrappers.
