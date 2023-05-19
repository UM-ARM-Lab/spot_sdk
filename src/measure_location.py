import argparse
import sys

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_se2_a_tform_b
from bosdyn.client.robot_state import RobotStateClient

def measure(config):
    bosdyn.client.util.setup_logging()
    sdk = bosdyn.client.create_standard_sdk("SpotClient")

    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    assert not robot.is_estopped(),"Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    state_client = robot.ensure_client(RobotStateClient.default_service_name)

    transforms = state_client.get_robot_state().kinematic_state.transforms_snapshot
    start_location_in_odom = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
    xo = start_location_in_odom.x
    yo = start_location_in_odom.y
    tho = start_location_in_odom.angle

    while True:

        transforms = state_client.get_robot_state().kinematic_state.transforms_snapshot

        body_pos_current = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
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