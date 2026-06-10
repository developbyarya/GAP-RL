import sapien
from sapien import pysapien
from sapien.physx import (
    PhysxArticulation,
    PhysxArticulationLinkComponent,
    PhysxContact,
    PhysxMaterial,
)
from sapien.render import RenderBodyComponent

sapien.Actor = pysapien.Entity
sapien.ActorBase = pysapien.Entity
sapien.Link = PhysxArticulationLinkComponent
sapien.LinkBase = PhysxArticulationLinkComponent
sapien.Articulation = PhysxArticulation
sapien.ArticulationBase = PhysxArticulation
sapien.Contact = PhysxContact
sapien.PhysicalMaterial = PhysxMaterial


def _entity_get_visual_bodies(self):
    return [c for c in self.get_components() if isinstance(c, RenderBodyComponent)]


pysapien.Entity.get_visual_bodies = _entity_get_visual_bodies


def _entity_hide_visual(self):
    for vb in self.get_visual_bodies():
        vb.visibility = 0.0


pysapien.Entity.hide_visual = _entity_hide_visual


def _entity_get_collision_shapes(self):
    from sapien.physx import PhysxRigidBaseComponent

    for c in self.get_components():
        if isinstance(c, PhysxRigidBaseComponent):
            return c.collision_shapes
    return []


pysapien.Entity.get_collision_shapes = _entity_get_collision_shapes


def _entity_get_id(self):
    return self.get_per_scene_id()


pysapien.Entity.get_id = _entity_get_id


def _entity_set_type(self, t):
    pass


def _entity_get_type(self):
    from sapien.physx import PhysxRigidDynamicComponent, PhysxRigidStaticComponent

    if self.find_component_by_type(PhysxRigidDynamicComponent):
        return "dynamic"
    if self.find_component_by_type(PhysxRigidStaticComponent):
        return "static"
    return "static"


pysapien.Entity.type = property(_entity_get_type, _entity_set_type)


def _entity_get_velocity(self):
    from sapien.physx import PhysxRigidDynamicComponent

    comp = self.find_component_by_type(PhysxRigidDynamicComponent)
    if comp is not None:
        return comp.linear_velocity
    return [0.0, 0.0, 0.0]


pysapien.Entity.get_velocity = _entity_get_velocity


def _entity_get_angular_velocity(self):
    from sapien.physx import PhysxRigidDynamicComponent

    comp = self.find_component_by_type(PhysxRigidDynamicComponent)
    if comp is not None:
        return comp.angular_velocity
    return [0.0, 0.0, 0.0]


pysapien.Entity.get_angular_velocity = _entity_get_angular_velocity


def _entity_set_velocity(self, vel):
    from sapien.physx import PhysxRigidDynamicComponent

    comp = self.find_component_by_type(PhysxRigidDynamicComponent)
    if comp is not None:
        comp.linear_velocity = vel


pysapien.Entity.set_velocity = _entity_set_velocity


def _entity_set_angular_velocity(self, vel):
    from sapien.physx import PhysxRigidDynamicComponent

    comp = self.find_component_by_type(PhysxRigidDynamicComponent)
    if comp is not None:
        comp.angular_velocity = vel


pysapien.Entity.set_angular_velocity = _entity_set_angular_velocity


PhysxArticulationLinkComponent.get_visual_bodies = lambda self: self.entity.get_visual_bodies()
PhysxArticulationLinkComponent.hide_visual = lambda self: self.entity.hide_visual()
PhysxArticulationLinkComponent.get_collision_shapes = lambda self: self.collision_shapes
PhysxArticulationLinkComponent.get_id = lambda self: self.entity.get_id()
