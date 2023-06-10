import argparse
import sys
import time

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient


def go_to_origin(config):
    bosdyn.client.util.setup_logging()
    sdk = bosdyn.client.create_standard_sdk("SpotClient")

    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.logger.info("Powering on robot... This may take several seconds.")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."
        robot.logger.info("Robot powered on.")

        robot.logger.info("Commanding robot to stand...")

        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        state_client = robot.ensure_client(RobotStateClient.default_service_name)

        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        # now give this to trajectory command to execute the transform
        body_move_command = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=0, goal_y=0, goal_heading=0, frame_name=ODOM_FRAME_NAME, locomotion_hint=spot_command_pb2.HINT_CRAWL)
        end_time = 5.0
        body_command_id = command_client.robot_command(body_move_command, end_time_secs=time.time() + end_time)
        time.sleep(3)

        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), "Robot power off failed."
        robot.logger.info("Robot safely powered off.")


def main(argv):
    parser = argparse.ArgumentParser()
    # adds hostname to parser
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)
    try:
        go_to_origin(options)
        return True
    except Exception as exc:
        logger = bosdyn.client.util.get_logger()
        logger.exception("Threw an exception")
        return False


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)
