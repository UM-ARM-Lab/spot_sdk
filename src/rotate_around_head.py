import argparse
import sys
import time
from math import pi, sin, cos
import matplotlib.pyplot as plt
import numpy as np

import go_to_point

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import arm_command_pb2, robot_command_pb2, geometry_pb2, world_object_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers, frame_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, HAND_FRAME_NAME, get_a_tform_b, get_se2_a_tform_b
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand, block_until_arm_arrives
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.power import PowerClient
from bosdyn.client.world_object import WorldObjectClient
from bosdyn import geometry

def rotate_around_arm(config):
    print("hello world")
    bosdyn.client.util.setup_logging()
    sdk = bosdyn.client.create_standard_sdk("SpotClient")

    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    assert not robot.is_estopped(),"Robot is estopped. Please use an external E-Stop client, " \
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
        power_client = robot.ensure_client(PowerClient.default_service_name)
        world_object_client = robot.ensure_client(WorldObjectClient.default_service_name)

        # response = power_client.fan_power_command(percent_power=1, duration=100)
        # time.sleep(1)
        # feedback1 = power_client.fan_power_command_feedback(response.command_id)
        
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        ##### Construct and grav aligned odom frame #####

        # frame_tree_edges = {}

        # vision_tform_special_frame

        ###### begin trajectory for deploying arm close to ground ####

        # Final position of the gripper relative to the gravity aligned body frmae
        x = 0.75
        y = 0
        z = -0.25

        hand_ewrt_flat_body = geometry_pb2.Vec3(x=x,y=y,z=z)

        # orientation of hand in rpy
        r = 0
        p = 1.25
        y = 0

        hand_rpy_euler = geometry.EulerZXY(roll=r,pitch=p,yaw=y)

        # Quaternion representation of rotation of hand in body frame
        hand_quat = hand_rpy_euler.to_quaternion()

        # SE3 of hand in flat body frame
        flat_body_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body, rotation=hand_quat)

        # Get the tf tree
        transforms = state_client.get_robot_state().kinematic_state.transforms_snapshot

        # body frame in odom
        odom_T_flat_body = get_a_tform_b(transforms, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

        # combine previously found transforms to express hand in odom frame
        # converts geometry_pb2.SE3 to math_helpers.pb2.SE3 
        odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_proto(flat_body_T_hand)

        seconds = 1.5
        
        arm_command = RobotCommandBuilder.arm_pose_command(odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w, odom_T_hand.rot.x,
            odom_T_hand.rot.y, odom_T_hand.rot.z, ODOM_FRAME_NAME, seconds)
        
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(0.7)

        # full synchronous arm and gripper cmd
        arm_deploy_command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)
        deploy_cmd_id = command_client.robot_command(arm_deploy_command)
        robot.logger.info("deploying arm.")

        block_until_arm_arrives(command_client, deploy_cmd_id)

        time.sleep(2)
        gripper_close_command = RobotCommandBuilder.claw_gripper_close_command()
        gripper_close_id = command_client.robot_command(gripper_close_command)
        robot.logger.info("closing gripper.")
        time.sleep(0.5)

        ## start rotating around a fixed point in space (the location of the hand)
        # move the body as intended and tell the gripper to synchronously go to the same location it started at in odom space
        
        # Update the "tf tree"
        transforms = state_client.get_robot_state().kinematic_state.transforms_snapshot

        # theta that the body must rotate in radians
        theta = pi

        traj_points, arm_points = calculate_traj_angle_points(r, p, transforms, theta)

        # # TODO: Don't forget to remove this eventually
        # arm_stow_command = RobotCommandBuilder.arm_stow_command()
        # arm_stow_id = command_client.robot_command(arm_stow_command)
        # time.sleep(0.5)

        # now give this to trajectory command to execute the transform
        body_poses = []
        arm_poses =[]
        for cmd_i in range(1,len(traj_points)):
            
            body_end_time = 4.5
            body_move_command = RobotCommandBuilder.synchro_se2_trajectory_command(
                goal_se2=traj_points[cmd_i].to_proto(), frame_name=ODOM_FRAME_NAME, locomotion_hint=spot_command_pb2.HINT_CRAWL)

            arm_end_time = 4
            arm_pos = arm_points[cmd_i]
            arm_command = RobotCommandBuilder.arm_pose_command(arm_pos.position.x, arm_pos.position.y, arm_pos.position.z, 
                                                               arm_pos.rotation.w, arm_pos.rotation.x, arm_pos.rotation.y, arm_pos.rotation.z, 
                                                               ODOM_FRAME_NAME, arm_end_time)
            
            body_command_id = command_client.robot_command(body_move_command, end_time_secs=time.time() + body_end_time)
            arm_command_id = command_client.robot_command(arm_command)

            transforms = state_client.get_robot_state().kinematic_state.transforms_snapshot
            body_pos_current = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
            arm_pos_current = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, HAND_FRAME_NAME)

            body_poses.append([body_pos_current.x, body_pos_current.y, body_pos_current.angle])
            arm_poses.append([arm_pos_current.x, arm_pos_current.y, arm_pos_current.angle])
            time.sleep(4.0)

        time.sleep(3)
        # planned_body_poses = [[p.x, p.y, p.angle] for p in traj_points]
        # plt.figure()
        # plt.scatter(arm_points[0].position.x, arm_points[0].position.y, label='hand')
        # plt.quiver([p[0] for p in planned_body_poses],
        #            [p[1] for p in planned_body_poses],
        #            [np.cos(p[2])*0.05 for p in planned_body_poses],
        #            [np.sin(p[2])*0.05 for p in planned_body_poses], label='planned')
        # plt.quiver([p[0] for p in body_poses],
        #            [p[1] for p in body_poses],
        #            [np.cos(p[2])*0.05 for p in body_poses],
        #            [np.sin(p[2])*0.05 for p in body_poses], label='estimated', color='b')
        # plt.axis("equal")
        # plt.show()
        # Maybe do a synchro arm and body command?
        # have an arm pos in the body frame now instead of odom frame
        drag_pos = math_helpers.SE2Pose(x=-1.5, y=0, angle=0)
        go_to_point.go_to_point(lease_client, robot, command_client, state_client, drag_pos)
        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), "Robot power off failed."
        robot.logger.info("Robot safely powered off.")
        
def calculate_traj_angle_points(hand_roll, hand_pitch, transforms, angle):
    # split the angle into ~ 30 degree angle chunks
    # Returns: a set of body waypoint poses in odom frame
    # TODO: Make a synchro arm command that also does this
    body_in_gripper = get_se2_a_tform_b(transforms, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME).inverse()

    traj_points = []
    poses = []
    arm_poses = []
    sub_angle = 0
    while (abs(sub_angle) < abs(angle)):
        traj_points.append(sub_angle)
        if angle < 0:
            sub_angle -= pi/6
        else:
            sub_angle += pi/6
    traj_points.append(angle)
    for a in traj_points:
        try:
            dx_body = cos(a) * body_in_gripper.x - sin(a) * body_in_gripper.y
            dy_body = sin(a) * body_in_gripper.x + cos(a) * body_in_gripper.y

            # math_helpers SE2 of change in body pose in hand frame, i.e. body in hand
            body_T_in_hand = math_helpers.SE2Pose.from_proto(geometry_pb2.SE2Pose(position=geometry_pb2.Vec2(x=dx_body, y=dy_body),angle=a))

            # SE2 describing body in odom frame by multiplying the grav_odom to grav_hand transform
            poses.append(get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, HAND_FRAME_NAME) * body_T_in_hand)

            # construct the quaternion for the hand position
            hand_pose_odom = get_a_tform_b(transforms, ODOM_FRAME_NAME, HAND_FRAME_NAME)
            current_hand_rot_odom = math_helpers.quat_to_eulerZYX(hand_pose_odom.rotation)
            # hand rpy relative to the odom frame
            hand_rpy_euler = geometry.EulerZXY(roll=hand_roll,pitch=hand_pitch,yaw=a+current_hand_rot_odom[0])
            hand_goal_odom = geometry_pb2.SE3Pose(position=hand_pose_odom.position, rotation=hand_rpy_euler.to_quaternion())
            arm_poses.append(hand_goal_odom)
        except:
            exit(1)
    return poses, arm_poses

def main(argv):
    parser = argparse.ArgumentParser()
    # adds hostname to parser
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)
    try:
        rotate_around_arm(options)
        return True
    except Exception as exc:
        logger = bosdyn.client.util.get_logger()
        logger.exception("Threw an exception")
        return False

if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)