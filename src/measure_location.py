import argparse
import sys
import time

import bosdyn.client
import bosdyn.client.util
from bosdyn.client import math_helpers, frame_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, geometry_pb2, get_se2_a_tform_b
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.world_object import WorldObjectClient, make_add_world_object_req
from bosdyn.api import world_object_pb2
from bosdyn.util import now_timestamp

def measure(config):
    bosdyn.client.util.setup_logging()
    sdk = bosdyn.client.create_standard_sdk("SpotClient")

    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    assert not robot.is_estopped(),"Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    state_client = robot.ensure_client(RobotStateClient.default_service_name)
    world_object_client = robot.ensure_client(WorldObjectClient.default_service_name)

    ### Construct map frame at the location of the robot when the program is run ###

    frame_tree_edges = {}
    map_frame = geometry_pb2.SE3Pose(position=geometry_pb2.Vec3(x=0,y=0,z=0),rotation=geometry_pb2.Quaternion(w=1,x=0,y=0,z=0))

    frame_tree_edges = frame_helpers.add_edge_to_tree(frame_tree_edges,map_frame,ODOM_FRAME_NAME,"map_frame")

    snapshot = geometry_pb2.FrameTreeSnapshot(child_to_parent_edge_map=frame_tree_edges)
    print("map_frame " , map_frame)
    world_obj_special_frame = world_object_pb2.WorldObject(id=21, name="MapFrame", transforms_snapshot=snapshot, acquisition_time=now_timestamp())

    world_object_client.mutate_world_objects(mutation_req=make_add_world_object_req(world_obj_special_frame))

    world_objects = world_object_client.list_world_objects().world_objects
    print("Current World objects before mutations: " + str([obj for obj in world_objects]))

    ## construct initial frame pos
    transforms = state_client.get_robot_state().kinematic_state.transforms_snapshot
    start_location_in_odom = get_se2_a_tform_b(transforms, "map_frame", GRAV_ALIGNED_BODY_FRAME_NAME)
    print(start_location_in_odom)
    xo = start_location_in_odom.x
    yo = start_location_in_odom.y
    tho = start_location_in_odom.angle

    while True:

        transforms = state_client.get_robot_state().kinematic_state.transforms_snapshot

        body_pos_current = get_se2_a_tform_b(transforms, "MapFrame", GRAV_ALIGNED_BODY_FRAME_NAME)
        print(body_pos_current)
        dispx = body_pos_current.x - xo
        dispy = body_pos_current.y - yo
        dispth = body_pos_current.angle - tho
        print("x: ", f'{dispx:.3f}', "y: ", f'{dispy:.3f}', "th: ", f'{dispth:.3f}')
        

def main(argv):
    parser = argparse.ArgumentParser()
    # adds hostname to parser
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)
    try:
        measure(options)
        return True
    except Exception as exc:
        logger = bosdyn.client.util.get_logger()
        logger.exception("Threw an exception")
        return False

if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)