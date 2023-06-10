import argparse
import io

from PIL import Image
import sys
import time
from math import pi

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
import cv2
import numpy as np
import rerun as rr
from bosdyn import geometry
from bosdyn.api import geometry_pb2, arm_command_pb2, synchronized_command_pb2, robot_command_pb2, \
    image_pb2, manipulation_api_pb2
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, get_a_tform_b, get_se2_a_tform_b
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient, blocking_stand,
                                         block_for_trajectory_cmd, block_until_arm_arrives)
from bosdyn.client.robot_state import RobotStateClient

g_image_click = None
g_image_display = None


def make_robot_command(arm_joint_traj):
    """ Helper function to create a RobotCommand from an ArmJointTrajectory.
        The returned command will be a SynchronizedCommand with an ArmJointMoveCommand
        filled out to follow the passed in trajectory. """

    joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=arm_joint_traj)
    arm_command = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=joint_move_command)
    sync_arm = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_command)
    arm_sync_robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_arm)
    return RobotCommandBuilder.build_synchro_command(arm_sync_robot_cmd)


def hand_gaze_ground_command(robot_state_client, x, y, z, roll=0, pitch=np.pi / 2, yaw=0, duration=0.5):
    """
    Move the arm to a pose relative to the body

    Args:
        robot_state_client: RobotStateClient
        x: x position in meters in front of the body center
        y: y position in meters to the left of the body center
        z: z position in meters above the body center
        roll: roll in radians
        pitch: pitch in radians
        yaw: yaw in radians
        duration: duration in seconds
    """
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

    hand_pos_in_body = geometry_pb2.Vec3(x=x, y=y, z=z)

    euler = geometry.EulerZXY(roll=roll, pitch=pitch, yaw=yaw)
    quat_hand = euler.to_quaternion()

    body_in_odom = get_a_tform_b(transforms, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
    hand_in_body = geometry_pb2.SE3Pose(position=hand_pos_in_body, rotation=quat_hand)

    hand_in_odom = body_in_odom * math_helpers.SE3Pose.from_proto(hand_in_body)

    arm_command = RobotCommandBuilder.arm_pose_command(
        hand_in_odom.x, hand_in_odom.y, hand_in_odom.z, hand_in_odom.rot.w, hand_in_odom.rot.x,
        hand_in_odom.rot.y, hand_in_odom.rot.z, ODOM_FRAME_NAME, duration)
    return arm_command


def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        # Draw some lines on the image.
        # print('mouse', x, y)
        color = (30, 30, 30)
        thickness = 2
        image_title = 'Click to grasp'
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone)


buffer = []


def force_measure(state_client, command_client):
    global buffer

    state = state_client.get_robot_state()
    manip_state = state.manipulator_state
    force_reading = manip_state.estimated_end_effector_force_in_hand
    total_force = np.sqrt(force_reading.x ** 2 + force_reading.y ** 2 + force_reading.z ** 2)

    # circular buffer
    buffer.append(total_force)
    if len(buffer) > 10:
        buffer.pop(0)
    recent_avg_total_force = float(np.mean(buffer))

    rr.log_scalar("force/x", force_reading.x)
    rr.log_scalar("force/y", force_reading.y)
    rr.log_scalar("force/z", force_reading.z)
    rr.log_scalar("force/total", total_force)
    rr.log_scalar("force/recent_avg_total", recent_avg_total_force)

    if recent_avg_total_force > 13:
        print("large force detected!", force_reading.z)
        stop_cmd = RobotCommandBuilder.stop_command()
        stop_cmd_id = command_client.robot_command(stop_cmd)
        return True
    return False


def drag_rope_to_pose(robot_state_client, command_client, se2pose, frame_name):
    # walk backwards
    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_command(goal_se2=se2pose.to_proto(), frame_name=frame_name,
                                                                   locomotion_hint=spot_command_pb2.HINT_CRAWL)
    walk_synchro_cmd = RobotCommandBuilder.build_synchro_command(robot_cmd)
    walk_cmd_id = command_client.robot_command(lease=None, command=walk_synchro_cmd,
                                               end_time_secs=time.time() + 999)

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
        if force_measure(robot_state_client, command_client):
            print("High force detected. Failed to reach goal.")
            return False
        time.sleep(0.25)


def arm_pickup(robot, robot_state_client, image_client, command_client, manipulation_api_client, disp_x=0.75, disp_y=0.,
               disp_z=0.2):
    look_ground_command = hand_gaze_ground_command(robot_state_client, disp_x, disp_y, disp_z)
    arm_cmd_id = command_client.robot_command(look_ground_command)
    block_until_arm_arrives(command_client, arm_cmd_id)

    # Take picture and get point
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
    g_image_click = None
    g_image_display = None
    cv2.destroyAllWindows()

    # build grasp

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

    # execute grasp
    t0 = time.time()
    while time.time() - t0 < 10:
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
        # Robot startup

        robot.logger.info("Powering on robot... This may take a several seconds.")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."
        robot.logger.info("Robot powered on.")

        robot.logger.info("Commanding robot to stand...")
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        initial_transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

        # pose to back up and see everything
        back_up_pose = pose_in_start_frame(initial_transforms, x=-1.5, y=0.4, angle=0)

        # goal pose
        goal_pose_in_odom = pose_in_start_frame(initial_transforms, x=0.75, y=-2.5, angle=pi / 2)

        # Pickup hose
        arm_pickup(robot=robot, image_client=image_client, manipulation_api_client=manipulation_api_client,
                   robot_state_client=robot_state_client, command_client=command_client)

        # start dragging hose toward the goal pose
        if not drag_rope_to_pose(robot_state_client=robot_state_client, command_client=command_client,
                                 se2pose=goal_pose_in_odom, frame_name=ODOM_FRAME_NAME):
            open_gripper_cmd = RobotCommandBuilder.claw_gripper_open_command()
            open_gripper_cmd_id = command_client.robot_command(open_gripper_cmd)
            # TODO: block until gripper is open?
            look_at_scene_cmd = hand_gaze_ground_command(robot_state_client, 0.7, 0, 0.2, roll=0, pitch=np.deg2rad(20),
                                                         yaw=0)
            look_at_scene_cmd_id = command_client.robot_command(look_at_scene_cmd)
            block_until_arm_arrives(command_client, look_at_scene_cmd_id)
            # FIXME: not sure why, but the arm status is "PROCESSING" after the above command
            #  even when it appears to have reached the goal, so we also send a stop command
            stop_cmd = RobotCommandBuilder.stop_command()
            command_client.robot_command(stop_cmd)

        # Regrasp

        # walk back to view the whole scene
        move_back_cmd = RobotCommandBuilder.synchro_se2_trajectory_command(goal_se2=back_up_pose.to_proto(),
                                                                           frame_name=ODOM_FRAME_NAME)
        move_back_cmd_id = command_client.robot_command(lease=None, command=move_back_cmd,
                                                        end_time_secs=time.time() + 999)
        block_body_trajectory_and_error(command_client, move_back_cmd_id)

        pil_images = get_images(image_client)

        for i, pil_image in enumerate(pil_images):
            pil_image.save(f"image{i}.png")

        # pickup hose
        arm_pickup(robot=robot, robot_state_client=robot_state_client, image_client=image_client,
                   command_client=command_client, manipulation_api_client=manipulation_api_client, disp_x=0.75,
                   disp_z=0.3)

        # # walk backward a little bit then put the hose down
        initial_transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
        put_down_hose_in_body = math_helpers.SE2Pose(x=0, y=-0.5, angle=0)
        put_down_hose_in_odom = get_se2_a_tform_b(initial_transforms, ODOM_FRAME_NAME,
                                                  GRAV_ALIGNED_BODY_FRAME_NAME) * put_down_hose_in_body
        put_down_hose_cmd = RobotCommandBuilder.synchro_se2_trajectory_command(
            goal_se2=put_down_hose_in_odom.to_proto(), frame_name=ODOM_FRAME_NAME,
            locomotion_hint=spot_command_pb2.HINT_CRAWL)

        lower_hand_cmd = hand_gaze_ground_command(robot_state_client, 0.75, 0.3, 0.2)

        unstick_hose_cmd = RobotCommandBuilder.build_synchro_command(put_down_hose_cmd, lower_hand_cmd)
        unstick_hose_cmd_id = command_client.robot_command(lease=None, command=lower_hand_cmd,
                                                           end_time_secs=time.time() + 4)

        time.sleep(2)
        open_gripper_cmd_stuck = RobotCommandBuilder.claw_gripper_open_command()
        gripper_open_command_id_2 = command_client.robot_command(open_gripper_cmd_stuck)
        time.sleep(2)
        # redrag

        arm_pickup(robot=robot, image_client=image_client, manipulation_api_client=manipulation_api_client,
                   robot_state_client=robot_state_client, command_client=command_client, disp_y=-0.3)
        time.sleep(4)
        drag_rope_to_pose(robot_state_client=robot_state_client, command_client=command_client,
                          se2pose=goal_pose_in_odom, frame_name=ODOM_FRAME_NAME)
        time.sleep(15)
        open_gripper_cmd_final = RobotCommandBuilder.claw_gripper_open_command()
        gripper_open_command_id = command_client.robot_command(open_gripper_cmd_final)

        # move to start pose
        move_to_start(command_client, initial_transforms)


def get_images(image_client):
    image_sources = [
        'hand_color_image',
        'frontleft_fisheye_image',
        'frontright_fisheye_image',
    ]
    rotations = [
        0,
        -90,
        -90,
    ]
    image_requests = [
        build_image_request(src, pixel_format=image_pb2.Image.PixelFormat.PIXEL_FORMAT_RGB_U8)
        for src in image_sources
    ]
    image_responses = image_client.get_image(image_requests)
    pil_images = [Image.open(io.BytesIO(res.shot.image.data)).convert('RGB').rotate(rot) for res, rot in
                  zip(image_responses, rotations)]
    return pil_images


def move_to_start(command_client, initial_transforms):
    start_pose_in_odom = pose_in_start_frame(initial_transforms, 0, 0, 0)
    move_to_start_cmd = RobotCommandBuilder.synchro_se2_trajectory_command(
        goal_se2=start_pose_in_odom.to_proto(), frame_name=ODOM_FRAME_NAME,
        locomotion_hint=spot_command_pb2.HINT_CRAWL)
    move_to_start_cmd_id = command_client.robot_command(lease=None, command=move_to_start_cmd,
                                                        end_time_secs=time.time() + 999)
    block_body_trajectory_and_error(command_client, move_to_start_cmd_id)


def pose_in_start_frame(initial_transforms, x, y, angle):
    """
    The start frame is where the robot starts.

    Args:
        initial_transforms: The initial transforms of the robot, this should be created at the beginning.
        x: The x position of the pose in the start frame.
        y: The y position of the pose in the start frame.
        angle: The angle of the pose in the start frame.
    """
    pose_in_body = math_helpers.SE2Pose(x=x, y=y, angle=angle)
    pose_in_odom = get_se2_a_tform_b(initial_transforms, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME) * pose_in_body
    return pose_in_odom


def block_body_trajectory_and_error(command_client, cmd_id):
    if not block_for_trajectory_cmd(command_client, cmd_id):
        raise RuntimeError()


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
    except Exception:
        logger = bosdyn.client.util.get_logger()
        logger.exception("Threw an exception")
        return False


if __name__ == '__main__':
    # Checks to see IP address is specified
    if not main(sys.argv[1:]):
        sys.exit(1)
