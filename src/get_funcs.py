import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image
from bosdyn.api import image_pb2, geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import get_a_tform_b, BODY_FRAME_NAME
from bosdyn.client.image import build_image_request, pixel_to_camera_space
from scipy import ndimage

from src.detect_regrasp_point import MODEL_VERSION, get_polys, DetectionError, hose_points_from_predictions, \
    detect_object_center, viz_detection, detect_regrasp_point_from_hose

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


@dataclass
class GetRetryResult:
    image_res: image_pb2.ImageResponse
    hose_points: np.ndarray
    best_idx: int
    best_vec2: geometry_pb2.Vec2


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


def np_to_vec2(a):
    return geometry_pb2.Vec2(x=a[0], y=a[1])


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
    img_str = base64.b64encode(cv2.imencode('.jpg', rgb_np)[1])
    upload_url = f"https://detect.roboflow.com/spot-vaccuming-demo/{MODEL_VERSION}?api_key={os.environ['ROBOFLOW_API_KEY']}"
    resp = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True).json()
    predictions = resp['predictions']
    return predictions


def save_data(rgb_np, depth_np, predictions):
    now = int(time.time())
    Path(f"data/{now}").mkdir(exist_ok=True, parents=True)

    Image.fromarray(rgb_np).save(f"data/{now}/rgb.png")
    Image.fromarray(np.squeeze(depth_np)).save(f"data/{now}/depth.png")
    with open(f"data/{now}/pred.json", 'w') as f:
        json.dump(predictions, f)


def get_mess(image_client):
    time.sleep(1)  # reduces motion blur?
    rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
    depth_np, depth_res = get_depth_img(image_client, 'hand_depth_in_hand_color_frame')

    predictions = get_predictions(rgb_np)

    save_data(rgb_np, depth_np, predictions)

    mess_polys = get_polys(predictions, "mess")

    if len(mess_polys) == 0:
        raise DetectionError("No mess detected")

    if len(mess_polys) != 1:
        print(f"Error: expected 1 mess, got {len(mess_polys)}")

    import matplotlib.pyplot as plt
    # Image.fromarray(rgb_np).save(f"mess_{int(time.time())}.png")
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

    depth_m = get_mess_depth(nonzero_depths_m)

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


def get_mess_depth(nonzero_depths_m):
    if len(nonzero_depths_m) > 0:
        depth_m = nonzero_depths_m.mean()
        if np.isfinite(depth_m):
            return depth_m
    return 3.5


def get_hose_and_head_point(image_client):
    time.sleep(1)  # reduces motion blur?
    rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
    depth_np, depth_res = get_depth_img(image_client, 'hand_depth_in_hand_color_frame')
    predictions = get_predictions(rgb_np)
    save_data(rgb_np, depth_np, predictions)

    hose_points = hose_points_from_predictions(predictions)
    head_detection = detect_object_center(predictions, "vacuum_head")

    fig, ax = viz_detection(rgb_np, head_detection)
    ax.plot(hose_points[:, 0], hose_points[:, 1], c='w', linewidth=4)
    fig.show()

    dists = np.linalg.norm(hose_points - head_detection.grasp_px, axis=-1)
    best_idx = int(np.argmin(dists))
    best_px = hose_points[best_idx]
    best_vec2 = np_to_vec2(best_px)

    ax.scatter(best_px[0], best_px[1], c='m', marker='*', s=100, zorder=10)
    fig.show()

    return GetRetryResult(rgb_res, hose_points, best_idx, best_vec2)


def get_hose_and_regrasp_point(image_client, ideal_dist_to_obs=50):
    time.sleep(1)  # reduces motion blur?
    rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
    depth_np, depth_res = get_depth_img(image_client, 'hand_depth_in_hand_color_frame')
    predictions = get_predictions(rgb_np)
    save_data(rgb_np, depth_np, predictions)

    hose_points = hose_points_from_predictions(predictions)

    best_idx, best_px = detect_regrasp_point_from_hose(rgb_np, predictions, ideal_dist_to_obs, hose_points)
    best_vec2 = np_to_vec2(best_px)
    return GetRetryResult(rgb_res, hose_points, best_idx, best_vec2)
