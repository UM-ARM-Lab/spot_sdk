import argparse
import sys
import time
from math import pi, sin, cos

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import arm_command_pb2, robot_command_pb2, geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, HAND_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand, block_until_arm_arrives
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.power import PowerClient
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

        # response = power_client.fan_power_command(percent_power=1, duration=100)
        # time.sleep(1)
        # feedback1 = power_client.fan_power_command_feedback(response.command_id)
        
        #blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        ###### begin trajectory for deploying arm close to ground ####

        # Final position of the gripper relative to the gravity aligned body frmae
        x = 0.75
        y = 0
        z = 0.25

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

        # Express this transform in odom frame
        robot_state = state_client.get_robot_state()

        # tf from odom to body frame
        odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

        # combine previously found transforms to express hand in odom frame
        # converts geometry_pb2.SE3 to math_helpers.pb2.SE3 
        odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_obj(flat_body_T_hand)

        seconds = 1.5
        
        arm_command = RobotCommandBuilder.arm_pose_command(odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w, odom_T_hand.rot.x,
            odom_T_hand.rot.y, odom_T_hand.rot.z, ODOM_FRAME_NAME, seconds)
        
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(0.7)

        # full synchronous arm and gripper cmd
        arm_deploy_command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)
        #deploy_cmd_id = command_client.robot_command(arm_deploy_command)
        robot.logger.info("deploying arm.")

        #block_until_arm_arrives(command_client, deploy_cmd_id)

        time.sleep(2)
        gripper_close_command = RobotCommandBuilder.claw_gripper_close_command()
        #gripper_close_id = command_client.robot_command(gripper_close_command)
        robot.logger.info("closing gripper.")
        time.sleep(0.5)

        ## start rotating around a fixed point in space (the location of the hand)
        # move the body as intended and tell the gripper to synchronously go to the same location it started at in odom space
        
        # get the "tf tree"
        transforms = state_client.get_robot_state().kinematic_state.transforms_snapshot

        # Somehow I think I have to set the arm frame?
        #  Transform in the arm frame
        # theta that the body must rotate in radians
        theta = pi / 3.0
        
        # body pose in hand frame
        body_pos_vector_in_hand = get_a_tform_b(transforms, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME).inverse()
        
        # body translation in hand frame
        # need to make a gravity aligned hand frame

        dx_body = cos(theta) * body_pos_vector_in_hand.x - sin(theta) * body_pos_vector_in_hand.y
        dy_body = sin(theta) * body_pos_vector_in_hand.x + cos(theta) * body_pos_vector_in_hand.y

        body_delta_T_translation = geometry_pb2.Vec3(x=dx_body, y=dy_body,z=body_pos_vector_in_hand.z)

        body_delta_T_euler_rot = geometry.EulerZXY(roll=0, pitch=0, yaw=theta)
        
        body_delta_T_quat = body_pos_vector_in_hand.rot * math_helpers.Quat.from_proto(body_delta_T_euler_rot.to_quaternion())
        body_T_euler_yaw = body_delta_T_quat.to_yaw()

        # hand_to_body
        body_T = geometry_pb2.SE3Pose(position=body_delta_T_translation, rotation=body_delta_T_quat.to_proto())

        # SE3 describing body in odom frame
        body_T_in_odom = get_a_tform_b(transforms, ODOM_FRAME_NAME, HAND_FRAME_NAME) * body_T

        # now give this to trajectory command somehow

        body_move_command = RobotCommandBuilder.trajectory_command(
            goal_x = body_delta_T_translation.x, goal_y = body_delta_T_translation.y, goal_heading=body_T_euler_yaw, frame_name=HAND_FRAME_NAME
        )
        end_time = 5.0
        gaze_target_in_odom = body_pos_vector_in_hand.transform_point(x = 1.5, y = 0, z = 0)
        gaze_command = RobotCommandBuilder.arm_gaze_command(gaze_target_in_odom[0],
                                                            gaze_target_in_odom[1],
                                                            gaze_target_in_odom[2],
                                                            ODOM_FRAME_NAME)
        
        #follow_arm_command = RobotCommandBuilder.follow_arm_command()

        #body_command_id = command_client.robot_command(body_move_command, end_time_secs=time.time() + end_time)
        #synchro_command = RobotCommandBuilder.build_synchro_command(body_move_command, gaze_command)
        #synchro_command_id = command_client.robot_command(synchro_command)
        #block_until_arm_arrives(command_client, synchro_command_id, 5.0)

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
        rotate_around_arm(options)
        return True
    except Exception as exc:
        logger = bosdyn.client.util.get_logger()
        logger.exception("Threw and exception")
        return False

if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)