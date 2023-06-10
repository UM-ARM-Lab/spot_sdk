import time

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, HAND_FRAME_NAME, get_a_tform_b, \
    get_se2_a_tform_b
from bosdyn.client.robot_command import RobotCommandBuilder, blocking_stand

import create_map_frame


def go_to_point(lease_client, robot, command_client, state_client, se2pose):
    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Construct "global" map frame
        transforms_map = create_map_frame.create_map_frame(robot, state_client)
        transforms = state_client.get_robot_state().kinematic_state.transforms_snapshot
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        # goal of body in map
        goal_in_body = math_helpers.SE2Pose(x=1.5, y=0, angle=0)
        goal_in_odom = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME) * goal_in_body
        end_time = 6.0
        body_move_command = RobotCommandBuilder.synchro_se2_trajectory_command(
            goal_se2=math_helpers.SE2Pose.to_proto(se2pose), frame_name=ODOM_FRAME_NAME,
            locomotion_hint=spot_command_pb2.HINT_CRAWL)
        hand_pos = get_a_tform_b(transforms, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME)
        arm_move_command = RobotCommandBuilder.arm_pose_command(x=hand_pos.position.x, y=hand_pos.position.y,
                                                                z=hand_pos.position.z, qw=hand_pos.rotation.w,
                                                                qx=hand_pos.rotation.x, qy=hand_pos.rotation.y,
                                                                qz=hand_pos.rotation.z,
                                                                frame_name=GRAV_ALIGNED_BODY_FRAME_NAME,
                                                                seconds=end_time)
        synchro_command = RobotCommandBuilder.build_synchro_command(arm_move_command, body_move_command)
        synchro_command_id = command_client.robot_command(synchro_command, end_time_secs=time.time() + end_time)

        # send arm trajectory too
        bring_arm_to_point(lease_client, robot, command_client, state_client, se2pose)
        time.sleep(9)


def bring_arm_to_point(lease_client, robot, command_client, state_client, pose_to_follow, end_time):
    assert not robot.is_estopped()
    transforms = state_client.get_root_state().kinematic_state.transforms_snapshot
    body_in_odom = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
    hand_in_body = get_a_tform_b(transforms, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME)

    # keep the current body to hand transform and just map it to a trajectory
    hand_move_command = RobotCommandBuilder.synchro_se2_trajectory_command * (goal_se2=math_helpers.SE2Pose.to)
