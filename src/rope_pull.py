import argparse
import base64
import json
import os
import sys
import time
from typing import List, Callable

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
import cv2
import numpy as np
import requests
import rerun as rr
from bosdyn import geometry
from bosdyn.api import geometry_pb2, arm_command_pb2, synchronized_command_pb2, robot_command_pb2, \
    image_pb2, manipulation_api_pb2
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, get_a_tform_b, get_se2_a_tform_b, \
    BODY_FRAME_NAME
from bosdyn.client.image import ImageClient, build_image_request, pixel_to_camera_space
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient, blocking_stand,
                                         block_for_trajectory_cmd, block_until_arm_arrives)
from bosdyn.client.robot_state import RobotStateClient
from google.protobuf import wrappers_pb2
from scipy import ndimage

from src.detect_regrasp_point import detect_object_center, viz_detection, DetectionError, get_polys, \
    detect_regrasp_point, MODEL_VERSION

FORCE_BUFFER_SIZE = 15

g_image_click = None
g_image_display = None

ROTATION_ANGLE = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontleft_depth_in_visual_frame': -78,
    'frontright_fisheye_image': -102,
    'frontright_depth_in_visual_frame': -102,
    'hand_depth_in_hand_color_frame': 0,
    'hand_depth': 0,
    'hand_color_image': 0,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180
}


def image_to_opencv(image, auto_rotate=True):
    """Convert an image proto message to an openCV image."""
    num_channels = 1  # Assume a default of 1 byte encodings.
    if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        dtype = np.uint16
    else:
        dtype = np.uint8
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            num_channels = 3
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
            num_channels = 4
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
            num_channels = 1
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
            num_channels = 1
            dtype = np.uint16

    img = np.frombuffer(image.shot.image.data, dtype=dtype)
    if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
        try:
            # Attempt to reshape array into a RGB rows X cols shape.
            img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_channels))
        except ValueError:
            # Unable to reshape the image data, trying a regular decode.
            img = cv2.imdecode(img, -1)
    else:
        img = cv2.imdecode(img, -1)

    if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
        img = img[:, :, ::-1]

    if auto_rotate:
        img = ndimage.rotate(img, ROTATION_ANGLE[image.source.name])

    return img


def blocking_arm_command(command_client, cmd):
    block_until_arm_arrives(command_client, command_client.robot_command(cmd))
    # FIXME: why is this needed???
    command_client.robot_command(RobotCommandBuilder.stop_command())


def block_for_manipulation_api_command(robot, manipulation_api_client, cmd_response):
    while True:
        time.sleep(0.25)
        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id)

        # Send the request
        response = manipulation_api_client.manipulation_api_feedback_command(
            manipulation_api_feedback_request=feedback_request)

        state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)
        robot.logger.info(f'Current state: {state_name}')

        if response.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
            break

    robot.logger.info('Finished.')


def make_robot_command(arm_joint_traj):
    """ Helper function to create a RobotCommand from an ArmJointTrajectory.
        The returned command will be a SynchronizedCommand with an ArmJointMoveCommand
        filled out to follow the passed in trajectory. """

    joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=arm_joint_traj)
    arm_command = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=joint_move_command)
    sync_arm = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_command)
    arm_sync_robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_arm)
    return RobotCommandBuilder.build_synchro_command(arm_sync_robot_cmd)


def look_at_command(robot_state_client, x, y, z, roll=0, pitch=np.pi / 2, yaw=0, duration=0.5):
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


def force_measure(state_client, command_client, force_buffer: List):
    state = state_client.get_robot_state()
    manip_state = state.manipulator_state
    force_reading = manip_state.estimated_end_effector_force_in_hand
    total_force = np.sqrt(force_reading.x ** 2 + force_reading.y ** 2 + force_reading.z ** 2)

    # circular buffer
    force_buffer.append(total_force)
    if len(force_buffer) > FORCE_BUFFER_SIZE:
        force_buffer.pop(0)
    recent_avg_total_force = float(np.mean(force_buffer))

    rr.log_scalar("force/x", force_reading.x)
    rr.log_scalar("force/y", force_reading.y)
    rr.log_scalar("force/z", force_reading.z)
    rr.log_scalar("force/total", total_force)
    rr.log_scalar("force/recent_avg_total", recent_avg_total_force)

    if recent_avg_total_force > 14 and len(force_buffer) == FORCE_BUFFER_SIZE:
        print(f"large force detected! {recent_avg_total_force:.2f}")
        command_client.robot_command(RobotCommandBuilder.stop_command())
        return True
    return False


def setup_and_stand(robot):
    robot.logger.info("Powering on robot... This may take a several seconds.")
    robot.power_on(timeout_sec=20)
    assert robot.is_powered_on(), "Robot power on failed."
    robot.logger.info("Robot powered on.")
    robot.logger.info("Commanding robot to stand...")
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    blocking_stand(command_client, timeout_sec=10)
    robot.logger.info("Robot standing.")
    return command_client


def np_to_vec2(a):
    return geometry_pb2.Vec2(x=a[0], y=a[1])


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


def look_at_scene(command_client, robot_state_client, x=0.75, y=0.1, z=0.4, pitch=0, yaw=0, dx=0., dy=0., dpitch=0.):
    look_cmd = look_at_command(robot_state_client, x + dx, y + dy, z,
                               0, pitch + dpitch, yaw,
                               duration=0.5)
    blocking_arm_command(command_client, look_cmd)


def get_color_img(image_client, src):
    rgb_req = build_image_request(src, pixel_format=image_pb2.Image.PixelFormat.PIXEL_FORMAT_RGB_U8)
    rgb_res = image_client.get_image([rgb_req])[0]
    rgb_np = image_to_opencv(rgb_res, auto_rotate=True)
    return rgb_np, rgb_res


def get_depth_img(image_client, src):
    depth_req = build_image_request(src, pixel_format=image_pb2.Image.PixelFormat.PIXEL_FORMAT_DEPTH_U16)
    depth_res = image_client.get_image([depth_req])[0]
    depth_np = image_to_opencv(depth_res, auto_rotate=True)
    return depth_np, depth_res


def get_predictions(rgb_np):
    def _get_predictions(_rgb_np):
        img_str = base64.b64encode(cv2.imencode('.jpg', _rgb_np)[1])
        upload_url = f"https://detect.roboflow.com/spot-vaccuming-demo/{MODEL_VERSION}?api_key={os.environ['ROBOFLOW_API_KEY']}"
        resp = requests.post(upload_url, data=img_str, headers={
            "Content-Type": "application/x-www-form-urlencoded"
        }, stream=True).json()
        _predictions = resp['predictions']
        return _predictions

    # FIXME: not sure why switching RGB<-->BGR seemed to fix it one time when I tried it
    predictions = _get_predictions(rgb_np)
    if len(predictions) == 0:
        predictions = get_predictions(rgb_np[:, :, ::-1])

    return predictions


def drag_rope_to_goal(robot_state_client, command_client, initial_transforms, x, y, angle):
    """
    Move the robot to a pose relative to the body while dragging the hose
    """
    force_buffer = []

    # Create the se2 trajectory for the dragging motion
    walk_cmd_id = walk_to_pose_in_initial_frame(command_client, initial_transforms, x=x, y=y, angle=angle,
                                                block=False)

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
        if force_measure(robot_state_client, command_client, force_buffer):
            print("High force detected. Failed to reach goal.")
            return False
        time.sleep(0.25)


def walk_to_pose_in_initial_frame(command_client, initial_transforms, x=0., y=0., angle=0., block=True):
    """
    Non-blocking, returns the command id
    """
    goal_pose_in_odom = pose_in_start_frame(initial_transforms, x=x, y=y, angle=angle)
    se2_cmd = RobotCommandBuilder.synchro_se2_trajectory_command(goal_se2=goal_pose_in_odom.to_proto(),
                                                                 frame_name=ODOM_FRAME_NAME,
                                                                 locomotion_hint=spot_command_pb2.HINT_CRAWL)
    se2_synchro_commnd = RobotCommandBuilder.build_synchro_command(se2_cmd)
    se2_cmd_id = command_client.robot_command(lease=None, command=se2_synchro_commnd, end_time_secs=time.time() + 999)
    if block:
        block_for_trajectory_cmd(command_client, se2_cmd_id)
    return se2_cmd_id


def rotate_image_coordinates(pts, width, height, rot):
    """
    Rotate image coordinates by rot degrees around the center of the image.

    Args:
        pts: Nx2 array of image coordinates
        width: width of image
        height: height of image
        rot: rotation in degrees
    """
    center = np.array([width / 2, height / 2])
    rot = np.deg2rad(rot)
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    new_pts = center + (pts - center) @ R.T
    return new_pts


def get_point_f_retry(command_client, robot_state_client, image_client, get_point_f: Callable,
                      y, z, pitch=0., yaw=0.):
    dx = 0
    dy = 0
    dpitch = 0
    while True:
        look_at_scene(command_client, robot_state_client, y=y, z=z, pitch=pitch, yaw=yaw, dx=dx, dy=dy, dpitch=dpitch)
        try:
            return get_point_f(image_client)
        except DetectionError:
            dx = np.random.randn() * 0.04
            dy = np.random.randn() * 0.04
            dpitch = np.random.randn() * 0.08


def get_regrasp_point(image_client):
    rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
    predictions = get_predictions(rgb_np)

    now = int(time.time())
    import json
    with open(f"pred_{now}.json", 'w') as f:
        json.dump(predictions, f)
    from PIL import Image
    pil_img = Image.fromarray(rgb_np)
    pil_img.save(f"img_{now}.png")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow(rgb_np)
    for pred in predictions:
        xs = [p['x'] for p in pred['points']]
        ys = [p['y'] for p in pred['points']]
        ax.plot(xs, ys)
    fig.show()

    regrasp_detection = detect_regrasp_point(predictions, 80)
    regrasp_vec = np_to_vec2(regrasp_detection.grasp_px)

    ax.scatter(regrasp_vec.x, regrasp_vec.y, s=100, marker='*', c='r', zorder=3)
    fig.show()

    return rgb_res, regrasp_vec


def get_mess(image_client):
    rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
    depth_np, depth_res = get_depth_img(image_client, 'hand_depth_in_hand_color_frame')

    predictions = get_predictions(rgb_np)

    mess_polys = get_polys(predictions, "mess")

    if len(mess_polys) == 0:
        raise DetectionError("No mess detected")

    if len(mess_polys) != 1:
        print(f"Error: expected 1 mess, got {len(mess_polys)}")

    from PIL import Image
    import matplotlib.pyplot as plt
    Image.fromarray(rgb_np).save(f"mess_{int(time.time())}.png")
    fig, ax = plt.subplots()
    ax.imshow(rgb_np, alpha=0.5, zorder=0)
    ax.imshow(depth_np, alpha=0.5, zorder=1)
    for mess_poly in mess_polys:
        ax.plot(mess_poly[:, 0], mess_poly[:, 1], zorder=2, linewidth=3)
    fig.show()

    # NOTE: we would need to handle the rotate if we used the body cameras
    mess_mask = np.zeros(depth_np.shape[:2])
    cv2.drawContours(mess_mask, mess_polys, -1, (1), 1)
    # expand the mask a bit
    mess_mask = cv2.dilate(mess_mask, np.ones((5, 5), np.uint8), iterations=1)
    depths_m = depth_np[np.where(mess_mask == 1)] / 1000
    nonzero_depths_m = depths_m[np.where(np.logical_and(depths_m > 0, np.isfinite(depths_m)))]
    depth_m = nonzero_depths_m.mean()

    if not np.isfinite(depth_m):
        raise DetectionError("depth is NaN")

    M = cv2.moments(mess_polys[0])
    mess_px = int(M["m10"] / M["m00"])
    mess_py = int(M["m01"] / M["m00"])

    mess_pos_in_cam = np.array(pixel_to_camera_space(rgb_res, mess_px, mess_py, depth=depth_m))  # [x, y, z]

    mess_in_cam = math_helpers.SE3Pose(*mess_pos_in_cam, math_helpers.Quat())
    # FIXME: why can't we use "GRAV_ALIGNED_BODY_FRAME_NAME" here?
    cam2body = get_a_tform_b(rgb_res.shot.transforms_snapshot, BODY_FRAME_NAME,
                             rgb_res.shot.frame_name_image_sensor)
    mess_in_body = cam2body * mess_in_cam
    mess_x = mess_in_body.x
    mess_y = mess_in_body.y

    print(f"Mess detected at {mess_x:.2f}, {mess_y:.2f}")
    return mess_x, mess_y


def get_vacuum_head_point(image_client):
    rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
    predictions = get_predictions(rgb_np)
    detection = detect_object_center(predictions, "vacuum_head")
    viz_detection(rgb_np, detection)
    from PIL import Image
    Image.fromarray(rgb_np).save("head.png")
    vacuum_head_vec = np_to_vec2(detection.grasp_px)
    return rgb_res, vacuum_head_vec


def walk_to_then_grasp(robot, robot_state_client, image_client, command_client, manipulation_api_client,
                       get_point_f: Callable):
    rgb_res, walk_vec = get_point_f_retry(command_client, robot_state_client, image_client, get_point_f,
                                          y=0.1,
                                          z=0.4,
                                          pitch=np.deg2rad(25))

    # NOTE: if we are going to use the body cameras, which are rotated, we also need to rotate the image coordinates
    # First just walk to in front of that point
    offset_distance = wrappers_pb2.FloatValue(value=1.00)
    walk_to_cmd = manipulation_api_pb2.WalkToObjectInImage(pixel_xy=walk_vec,
                                                           transforms_snapshot_for_camera=rgb_res.shot.transforms_snapshot,
                                                           frame_name_image_sensor=rgb_res.shot.frame_name_image_sensor,
                                                           camera_model=rgb_res.source.pinhole,
                                                           offset_distance=offset_distance)
    walk_to_request = manipulation_api_pb2.ManipulationApiRequest(walk_to_object_in_image=walk_to_cmd)
    walk_response = manipulation_api_client.manipulation_api_command(manipulation_api_request=walk_to_request)
    block_for_manipulation_api_command(robot, manipulation_api_client, walk_response)

    rgb_res, pick_vec = get_point_f_retry(command_client, robot_state_client, image_client, get_point_f,
                                          z=0.45,
                                          y=0.1,
                                          pitch=1.25)
    do_grasp(robot, manipulation_api_client, rgb_res, pick_vec)


def do_grasp(robot, manipulation_api_client, image_res, pick_vec):
    pick_cmd = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=pick_vec, transforms_snapshot_for_camera=image_res.shot.transforms_snapshot,
        frame_name_image_sensor=image_res.shot.frame_name_image_sensor,
        camera_model=image_res.source.pinhole)
    grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=pick_cmd)
    cmd_response = manipulation_api_client.manipulation_api_command(manipulation_api_request=grasp_request)

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

    # Video recording
    # device_num = 4
    # cap = cv2.VideoCapture(device_num)
    # vr = VideoRecorder(cap, 'video/')
    # vr.start_new_recording(f'demo_{int(time.time())}.mp4')
    # vr.start_in_thread()

    arm_pull_rope_with_lease(lease_client, image_client, manipulation_api_client, robot, robot_state_client)

    # vr.stop_in_thread()


def arm_pull_rope_with_lease(lease_client, image_client, manipulation_api_client, robot, robot_state_client):
    with (bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)):
        command_client = setup_and_stand(robot)

        initial_transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

        # open the hand, so we can see more with the depth sensor
        command_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())

        # first detect the goal
        mess_x, mess_y = get_point_f_retry(command_client, robot_state_client, image_client, get_mess,
                                           y=-0.1, z=0.5,
                                           pitch=np.deg2rad(20), yaw=-0.7)

        # Grasp the hose to DRAG
        walk_to_then_grasp(robot, robot_state_client, image_client, command_client, manipulation_api_client,
                           get_vacuum_head_point)

        goal_reached = drag_rope_to_goal(robot_state_client, command_client, initial_transforms, mess_x, mess_y,
                                         np.pi / 2)
        time.sleep(1)  # makes the video look better in my opinion
        if goal_reached:
            robot.logger.info("Goal reached!")
            return

        command_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
        # FIXME: block until gripper is open?
        blocking_arm_command(command_client, RobotCommandBuilder.arm_ready_command())

        # Regrasp
        walk_to_pose_in_initial_frame(command_client, initial_transforms, x=0.0, y=0.0, angle=0.0)

        # Grasp the hose to get it UNSTUCK
        walk_to_then_grasp(robot, robot_state_client, image_client, command_client, manipulation_api_client,
                           get_regrasp_point)

        # Move the arm to get the hose unstuck
        blocking_arm_command(command_client, look_at_command(robot_state_client, 1.0, 0, 0.2))
        blocking_arm_command(command_client, look_at_command(robot_state_client, 1.0, -0.45, 0.2))
        blocking_arm_command(command_client, look_at_command(robot_state_client, 1.0, -0.45, -0.4))

        # Open the gripper
        command_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
        time.sleep(1)  # FIXME: how to block on a gripper command?

        # Stow
        blocking_arm_command(command_client, RobotCommandBuilder.arm_stow_command())

        # Look at the scene
        walk_to_pose_in_initial_frame(command_client, initial_transforms, x=-0.1, y=-0.1, angle=0)

        # Grasp the hose to DRAG again
        walk_to_then_grasp(robot, robot_state_client, image_client, command_client, manipulation_api_client,
                           get_vacuum_head_point)

        # try again to drag the hose to the goal
        goal_reached = drag_rope_to_goal(robot_state_client, command_client, initial_transforms, mess_x, mess_y,
                                         np.pi / 2)
        robot.logger.info("Goal reached: %s", goal_reached)

        # TODO: rotate around hand?
        # This is sort of cheating, since we're not monitoring force
        # walk_to_pose_in_initial_frame(command_client, initial_transforms, x=3, y=-1.9, angle=0, block=True)

        input("Press enter to finish")
        return


def main(argv):
    pass
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)
    rr.init("rope_pull")
    rr.connect()
    arm_pull_rope(options)


if __name__ == '__main__':
    # Checks to see IP address is specified
    if not main(sys.argv[1:]):
        sys.exit(1)
