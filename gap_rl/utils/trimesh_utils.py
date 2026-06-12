from typing import Dict, List

import numpy as np
import sapien
import trimesh


def get_actor_meshes(actor: "sapien.ActorBase"):
    """Get actor (collision) meshes in the actor frame."""
    meshes = []
    for col_shape in actor.get_collision_shapes():
        if isinstance(col_shape, sapien.physx.PhysxCollisionShapeBox):
            mesh = trimesh.creation.box(extents=2 * np.array(col_shape.half_size))
        elif isinstance(col_shape, sapien.physx.PhysxCollisionShapeCapsule):
            mesh = trimesh.creation.capsule(
                height=2 * col_shape.half_length, radius=col_shape.radius
            )
        elif isinstance(col_shape, sapien.physx.PhysxCollisionShapeSphere):
            mesh = trimesh.creation.icosphere(radius=col_shape.radius)
        elif isinstance(col_shape, sapien.physx.PhysxCollisionShapePlane):
            continue
        elif isinstance(
            col_shape,
            (sapien.physx.PhysxCollisionShapeConvexMesh, sapien.physx.PhysxCollisionShapeTriangleMesh),
        ):
            vertices = col_shape.vertices  # [n, 3]
            faces = col_shape.triangles  # [m, 3]
            vertices = vertices * col_shape.scale
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        else:
            raise TypeError(type(col_shape))
        mesh.apply_transform(col_shape.get_local_pose().to_transformation_matrix())
        meshes.append(mesh)
    return meshes


def get_visual_body_meshes(visual_body: "sapien.render.RenderBodyComponent"):
    meshes = []
    for render_shape in visual_body.render_shapes:
        for part in render_shape.get_parts():
            vertices = part.vertices  # [n, 3]
            faces = part.triangles  # [m, 3]
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh.apply_transform(visual_body.pose.to_transformation_matrix())
            meshes.append(mesh)
    return meshes


def get_actor_visual_meshes(actor: "sapien.ActorBase"):
    """Get actor (visual) meshes in the actor frame."""
    meshes = []
    for comp in actor.entity.components:
        if isinstance(comp, sapien.render.RenderBodyComponent):
            meshes.extend(get_visual_body_meshes(comp))
    return meshes


def merge_meshes(meshes: List[trimesh.Trimesh]):
    n, vs, fs = 0, [], []
    for mesh in meshes:
        v, f = mesh.vertices, mesh.faces
        vs.append(v)
        fs.append(f + n)
        n = n + v.shape[0]
    if n:
        return trimesh.Trimesh(np.vstack(vs), np.vstack(fs))
    else:
        return None


def get_actor_mesh(actor: "sapien.ActorBase", to_world_frame=True):
    mesh = merge_meshes(get_actor_meshes(actor))
    if mesh is None:
        return None
    if to_world_frame:
        T = actor.pose.to_transformation_matrix()
        mesh.apply_transform(T)
    return mesh


def get_actor_visual_mesh(actor: "sapien.ActorBase", to_world_frame=True):
    mesh = merge_meshes(get_actor_visual_meshes(actor))
    if mesh is None:
        return None
    if to_world_frame:
        T = actor.pose.to_transformation_matrix()
        mesh.apply_transform(T)
    return mesh


def get_articulation_meshes(
    articulation: "sapien.ArticulationBase", include_link_names=(), exclude_link_names=()
):
    """Get link meshes in the world frame."""
    meshes = []
    for link in articulation.get_links():
        if link.name in exclude_link_names:
            continue
        if not include_link_names or link.name in include_link_names:
            mesh = get_actor_mesh(link, True)
            if mesh is not None:
                meshes.append(mesh)
    return meshes


def get_articulation_visual_meshes(
    articulation: "sapien.ArticulationBase", include_link_names=(), exclude_link_names=()
):
    """Get link meshes in the world frame."""
    meshes = []
    for link in articulation.get_links():
        if link.name in exclude_link_names:
            continue
        if not include_link_names or link.name in include_link_names:
            mesh = get_actor_visual_mesh(link, True)
            if mesh is not None:
                meshes.append(mesh)
    return meshes
