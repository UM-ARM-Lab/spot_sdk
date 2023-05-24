import argparse
import sys
import time
from math import sqrt

import bosdyn.client
import bosdyn.client.util
from bosdyn.client import math_helpers, frame_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, BODY_FRAME_NAME, ODOM_FRAME_NAME, geometry_pb2, get_se2_a_tform_b, get_a_tform_b
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.world_object import WorldObjectClient, make_add_world_object_req
from bosdyn.api import world_object_pb2
from bosdyn.util import now_timestamp

def create_map_frame(robot, state_client):
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    assert not robot.is_estopped(),"Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    world_object_client = robot.ensure_client(WorldObjectClient.default_service_name)

    ### Construct map frame at the location of the robot when the program is run ###

    transforms = state_client.get_robot_state().kinematic_state.transforms_snapshot
    # Get pose of body in odom to invert and zero it in the map frame
    body_in_odom = get_a_tform_b(transforms, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

    frame_tree_edges = {}
    map_frame = geometry_pb2.SE3Pose(position=(-math_helpers.Vec3.from_proto(body_in_odom.position)).to_proto(), rotation=math_helpers.Quat.from_proto(body_in_odom.rotation).inverse().to_proto())

    frame_tree_edges = frame_helpers.add_edge_to_tree(frame_tree_edges,map_frame,ODOM_FRAME_NAME,"map_frame")

    snapshot = geometry_pb2.FrameTreeSnapshot(child_to_parent_edge_map=frame_tree_edges)
    world_obj_special_frame = world_object_pb2.WorldObject(id=21, name="MapFrame", transforms_snapshot=snapshot, acquisition_time=now_timestamp())

    world_object_client.mutate_world_objects(mutation_req=make_add_world_object_req(world_obj_special_frame))
    world_objects = world_object_client.list_world_objects().world_objects

    ## construct initial frame pos
    transforms_map = state_client.get_robot_state().kinematic_state.transforms_snapshot

    for world_obj in world_objects:
        if world_obj.name == "MapFrame":
            transforms_map = world_obj.transforms_snapshot
    return transforms_map
