import argparse
import sys
import time

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.api import geometry_pb2, trajectory_pb2, arm_command_pb2, synchronized_command_pb2, robot_command_pb2, image_pb2, manipulation_api_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2

from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, get_a_tform_b, get_se2_a_tform_b
from bosdyn.client import math_helpers

from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient, blocking_stand, block_until_arm_arrives)
from bosdyn.util import seconds_to_duration

from bosdyn import geometry

import rerun as rr
import numpy as np
import cv2
from collections import deque

# global forceQueue
# global runningSum
g_image_click = None
g_image_display = None


def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        # Draw some lines on the image.
        #print('mouse', x, y)
        color = (30, 30, 30)
        thickness = 2
        image_title = 'Click to grasp'
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone)

def force_measure(state_client):
    state = state_client.get_robot_state()
    manip_state = state.manipulator_state
    force_reading = manip_state.estimated_end_effector_force_in_hand
    rr.log_scalar("force/x", force_reading.x)
    rr.log_scalar("force/y", force_reading.y)
    rr.log_scalar("force/z", force_reading.z)
    # runningSum = runningSum - forceQueue.popleft() + force_reading.z
    # forceQueue.append(force_reading.z)
    if abs(force_reading.z) > 15:
        # print(runningSum)
        print("large z force detected!",force_reading.z)
        return True
    return False

def drag_rope_to_pose(robot, robot_state_client, command_client, se2pose, frame_name):
        transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

        x = 0.75
        y = 0
        z = -0.1
        r = 0
        p = 1.57
        y = 0

        hand_pos_in_body = geometry_pb2.Vec3(x=x,y=y,z=z)

        euler = geometry.EulerZXY(roll=r,pitch=p,yaw=y)
        quat_hand=euler.to_quaternion()

        body_in_odom = get_a_tform_b(transforms, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
        hand_in_body = geometry_pb2.SE3Pose(position=hand_pos_in_body, rotation=quat_hand)

        hand_in_odom = body_in_odom * math_helpers.SE3Pose.from_proto(hand_in_body)

        # duration in seconds
        seconds = 0.5

        arm_command = RobotCommandBuilder.arm_pose_command(
            hand_in_odom.x, hand_in_odom.y, hand_in_odom.z, hand_in_odom.rot.w, hand_in_odom.rot.x,
            hand_in_odom.rot.y, hand_in_odom.rot.z, ODOM_FRAME_NAME, seconds)

        command = RobotCommandBuilder.build_synchro_command(arm_command)
        arm_cmd_id = command_client.robot_command(command)
        time.sleep(0.5)

        # lock hand in body until the desired pose is reached or the force sensor detects a high force

        robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_command(goal_se2=se2pose.to_proto(), frame_name=frame_name, locomotion_hint=spot_command_pb2.HINT_CRAWL)
        end_time = 10
        walk_cmd_id = command_client.robot_command(lease=None, command=robot_cmd, end_time_secs=time.time() + end_time)

        # loop to check forces
        while True:
            feedback = command_client.robot_command_feedback(walk_cmd_id)
            mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
            if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
                print("Failed to reach goal.")
                return False
            traj_feedback = mobility_feedback.se2_trajectory_feedback
            if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                    traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
                print("Arrived at goal.")
                return True
            if (force_measure(robot_state_client)):
                print("High force detected. Failed to reach goal.")
                return False
            time.sleep(0.25)   

def arm_pickup(robot, robot_state_client, image_client, command_client, manipulation_api_client):
        transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

        x = 0.75
        y = 0
        z = -0.1
        r = 0
        p = 1.3
        y = 0

        hand_pos_in_body = geometry_pb2.Vec3(x=x,y=y,z=z)

        euler = geometry.EulerZXY(roll=r,pitch=p,yaw=y)
        quat_hand=euler.to_quaternion()

        body_in_odom = get_a_tform_b(transforms, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
        hand_in_body = geometry_pb2.SE3Pose(position=hand_pos_in_body, rotation=quat_hand)

        hand_in_odom = body_in_odom * math_helpers.SE3Pose.from_proto(hand_in_body)

        # duration in seconds
        seconds = 0.5

        arm_command = RobotCommandBuilder.arm_pose_command(
            hand_in_odom.x, hand_in_odom.y, hand_in_odom.z, hand_in_odom.rot.w, hand_in_odom.rot.x,
            hand_in_odom.rot.y, hand_in_odom.rot.z, ODOM_FRAME_NAME, seconds)

        command = RobotCommandBuilder.build_synchro_command(arm_command)
        arm_cmd_id = command_client.robot_command(command)
        time.sleep(2)

        ### Take picture and get point ###

        robot.logger.info('Getting an image from: hand_color_image')
        image_responses = image_client.get_image_from_sources(["hand_color_image"])

        if len(image_responses) != 1:
            print('Got invalid number of images: ' + str(len(image_responses)))
            print(image_responses)
            assert False
        
        image = image_responses[0]
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16
        else:
            dtype = np.uint8
        img = np.fromstring(image.shot.image.data, dtype=dtype)
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            img = img.reshape(image.shot.image.rows, image.shot.image.cols)
        else:
            img = cv2.imdecode(img, -1)

        robot.logger.info('Click on an object to start grasping...')
        image_title = 'Click to grasp'
        cv2.namedWindow(image_title)
        cv2.setMouseCallback(image_title, cv_mouse_callback)

        global g_image_click, g_image_display
        g_image_display = img
        cv2.imshow(image_title, g_image_display)
        while g_image_click is None:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                # Quit
                print('"q" pressed, exiting.')
                exit(0)
        
        robot.logger.info('Picking object at image location (' + str(g_image_click[0]) + ', ' +
                          str(g_image_click[1]) + ')')

        pick_vec = geometry_pb2.Vec2(x=g_image_click[0], y=g_image_click[1])

        #### build grasp ####

        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec, transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=image.source.pinhole)

        # Add squeeze grasp constraint

        grasp.grasp_params.grasp_params_frame_name = ODOM_FRAME_NAME

        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.squeeze_grasp.SetInParent()

        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)
        cmd_response = manipulation_api_client.manipulation_api_command(
            manipulation_api_request=grasp_request)
        
        ### execute grasp ###

        while True:
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)

            # Send the request
            response = manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)

            print('Current state: ',
                  manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state))

            if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                break

            time.sleep(1)

        robot.logger.info('Finished grasp.')
            
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
    image_client = robot.ensure_client(ImageClient.default_service_name)

    lease_client.take()

    with (bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)):

        #### Robot startup ####

        robot.logger.info("Powering on robot... This may take a several seconds.")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."
        robot.logger.info("Robot powered on.")

        robot.logger.info("Commanding robot to stand...")
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

        #### poses for all the points ####
        # initial hose position (1)
        hose_start_in_body = math_helpers.SE2Pose(x=1.2,y=0,angle=0)
        hose_start_in_odom = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME) * hose_start_in_body

        # pose to unstick
        stuck_pose_in_body = math_helpers.SE2Pose(x=2.5,y=0,angle=0)
        stuck_pose_in_odom = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME,GRAV_ALIGNED_BODY_FRAME_NAME) * stuck_pose_in_body

        # goal pose
        goal_pose_in_body = math_helpers.SE2Pose(x=-0.7, y=-0.5,angle=0)
        goal_pose_in_odom = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME) * goal_pose_in_body

        end_time = 4.0

        body_move_to_hose_start_cmd = RobotCommandBuilder.synchro_se2_trajectory_command(goal_se2=hose_start_in_odom.to_proto(), frame_name=ODOM_FRAME_NAME, locomotion_hint= spot_command_pb2.HINT_CRAWL)
        # move_to_hose_id = command_client.robot_command(body_move_to_hose_start_cmd, end_time_secs=time.time() + end_time)
        
        # time.sleep(end_time + 1.5)

        #### Pickup hose ###
        arm_pickup(robot=robot, robot_state_client=robot_state_client, image_client=image_client, command_client=command_client, manipulation_api_client=manipulation_api_client)

        ### start dragging hose toward the goal pose ###

        drag_rope_to_pose(robot, robot_state_client,command_client, goal_pose_in_odom, ODOM_FRAME_NAME)

        # Open the gripper
        gripper_open = RobotCommandBuilder.claw_gripper_open_fraction_command(0.5)
        gripper_open_command_id = command_client.robot_command(gripper_open)

        robot.logger.info("Open gripper command issued.")
        block_until_arm_arrives(command_client, gripper_open_command_id, 3.0)
        
        # Wait 2 seconds to let a user put in a rope or something like that
        time.sleep(2.0)

def main(argv):
    pass
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)
    rr.init("rope_pull")
    rr.connect()
    try:
        # runningSum = 0
        # forceQueue.append(0.0)
        # forceQueue.append(0.0)
        # forceQueue.append(0.0)
        # forceQueue.append(0.0)
        # forceQueue.append(0.0)
        # forceQueue = deque()
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
