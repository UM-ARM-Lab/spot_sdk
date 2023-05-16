import argparse
import sys
import time

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import geometry_pb2, trajectory_pb2, arm_command_pb2, synchronized_command_pb2, robot_command_pb2

from bosdyn.client.frame_helpers import ODOM_FRAME_NAME

from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient, blocking_stand, block_until_arm_arrives)
from bosdyn.util import seconds_to_duration

import rerun as rr

def arm_pull_rope(config):
    """Applies force to the gripper so it remains stationary. If force surpasses a certain limit, flash the LEDs red"""

    bosdyn.client.util.setup_logging(config.verbose)

    # Creates client, robot, and authenticates, and time syncs
    sdk = bosdyn.client.create_standard_sdk('RopePullClient')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), "Robot requires an arm to run this example."

    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, such as the" \
        " estop SDK example, to configure E-Stop."

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)

    with (bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)):
        robot.logger.info("Powering on robot... This may take a several seconds.")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."
        robot.logger.info("Robot powered on.")

        robot.logger.info("Commanding robot to stand...")
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        # Unstow the arm
        unstow = RobotCommandBuilder.arm_ready_command()
        unstow_command_id = command_client.robot_command(unstow)

        robot.logger.info("Unstow command issued.")
        block_until_arm_arrives(command_client, unstow_command_id, 3.0)

        # Open the gripper
        gripper_open = RobotCommandBuilder.claw_gripper_open_fraction_command(0.5)
        gripper_open_command_id = command_client.robot_command(gripper_open)

        robot.logger.info("Open gripper command issued.")
        block_until_arm_arrives(command_client, gripper_open_command_id, 3.0)
        
        # Wait 2 seconds to let a user put in a rope or something like that
        time.sleep(2.0)

        # Close the gripper
        gripper_close = RobotCommandBuilder.claw_gripper_open_fraction_command(0)
        gripper_close_command_id = command_client.robot_command(gripper_close)
        
        robot.logger.info("Close gripper command issued.")
        block_until_arm_arrives(command_client, gripper_close_command_id, 3.0)        

        # ___________

        # Hold arm in place with a specified force while measuring applied force on EE
        # Using a wrench trajectory?

        f_x0 = -5
        f_y0 = 0
        f_z0 = 0

        f_x1 = -5
        f_y1 = 0
        f_z1 = 0

        torque_x = 0
        torque_y = 0
        torque_z = 0

        trajectory_duration = 10

        # First point in trajectory
        force0 = geometry_pb2.Vec3(x=f_x0, y =f_y0, z=f_z0)
        torque0 = geometry_pb2.Vec3(x=torque_x, y=torque_y, z=torque_z)

        wrench0 = geometry_pb2.Wrench(force=force0, torque=torque0)
        t0 = seconds_to_duration(0)
        traj_point0 = trajectory_pb2.WrenchTrajectoryPoint(wrench=wrench0, time_since_reference=t0)

        #Second point in trajectory
        force1 = geometry_pb2.Vec3(x=f_x1, y =f_y1, z=f_z1)
        torque1 = geometry_pb2.Vec3(x=torque_x, y=torque_y, z=torque_z)        

        wrench1 = geometry_pb2.Wrench(force=force1, torque=torque1)
        t1 = seconds_to_duration(trajectory_duration)
        traj_point1 = trajectory_pb2.WrenchTrajectoryPoint(wrench=wrench1, time_since_reference=t1)

        # Build trajectory
        trajectory = trajectory_pb2.WrenchTrajectory(points=[traj_point0, traj_point1])
        
        # Construct cmd request

        arm_cartesian_command = arm_command_pb2.ArmCartesianCommand.Request(
            root_frame_name=ODOM_FRAME_NAME,
            wrench_trajectory_in_task=trajectory,
            x_axis=arm_command_pb2.ArmCartesianCommand.Request.AXIS_MODE_FORCE,
            y_axis=arm_command_pb2.ArmCartesianCommand.Request.AXIS_MODE_FORCE,
            z_axis=arm_command_pb2.ArmCartesianCommand.Request.AXIS_MODE_FORCE,
            rx_axis=arm_command_pb2.ArmCartesianCommand.Request.AXIS_MODE_FORCE,
            ry_axis=arm_command_pb2.ArmCartesianCommand.Request.AXIS_MODE_FORCE,
            rz_axis=arm_command_pb2.ArmCartesianCommand.Request.AXIS_MODE_FORCE)
        
        arm_command = arm_command_pb2.ArmCommand.Request(arm_cartesian_command=arm_cartesian_command)
        synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(
            arm_command=arm_command)
        robot_command = robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)

        # Send request
        command_client.robot_command(robot_command)
        robot.logger.info('Force trajectory command issued... ')

        t_end = time.time() + 3 + trajectory_duration
        while time.time() < t_end:
            state = robot_state_client.get_robot_state()
            manip_state = state.manipulator_state
            force_reading = manip_state.estimated_end_effector_force_in_hand
            rr.log_scalar("force/x", force_reading.x)
            rr.log_scalar("force/y", force_reading.y)
            rr.log_scalar("force/z", force_reading.z)
            print(manip_state.estimated_end_effector_force_in_hand)

        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), "Robot power off failed."
        robot.logger.info("Robot safely powered off.")
            


def main(argv):
    pass
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)
    rr.init("rope_pull")
    rr.connect()
    try:
        arm_pull_rope(options)
        return True
    except Exception as exc:
        logger = bosdyn.client.util.get_logger()
        logger.exception("Threw an exception")
        return False


if __name__ == '__main__':
    # Checks to see IP address is specified
    if not main(sys.argv[1:]):
        sys.exit(1)
