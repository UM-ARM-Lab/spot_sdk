# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Example demonstrating capture of both visual and depth images and then overlaying them."""

import argparse
import sys

import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient
from PIL import Image
import io

import time

def main(argv):
    # Parse args
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--to-depth',
                        help='Convert to the depth frame. Default is convert to visual.',
                        action='store_true')
    parser.add_argument('--camera', help='Camera to acquire image from.', default='frontleft',\
                        choices=['frontleft', 'frontright', 'left', 'right', 'back',
                        ])
    parser.add_argument('--auto-rotate', help='rotate right and front images to be upright',
                        action='store_true')
    options = parser.parse_args(argv)

    if options.to_depth:
        sources = [options.camera + '_depth', options.camera + '_visual_in_depth_frame']
    else:
        sources = [options.camera + '_depth_in_visual_frame', options.camera + '_fisheye_image']

    sources = ['frontleft_fisheye_image', 'hand_color_image']
    # Create robot object with an image client.
    sdk = bosdyn.client.create_standard_sdk('image_depth_plus_visual')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    
    counter = 60
    while True:
        # Capture and save images to disk
        x = image_client.list_image_sources()
        image_responses = image_client.get_image_from_sources(sources)

        # Image responses are in the same order as the requests.
        # Convert to opencv images.
        # Visual is a JPEG
        image_responses = image_client.get_image_from_sources(sources)
        # get binary pillow image
        pil_image_body = Image.open(io.BytesIO(image_responses[0].shot.image.data)).convert('RGB')
        pil_image_hand = Image.open(io.BytesIO(image_responses[1].shot.image.data)).convert('RGB')
        # convert to cv mat
        cv_color_image_hand = np.array(pil_image_hand)
        cv_color_image_body = np.array(pil_image_body)
        # fix color to be BGR
        cv_color_image_hand = cv_color_image_hand[:, :, ::-1].copy()
        cv_color_image_body = cv_color_image_body[:, :, ::-1].copy()
        
        cv_visual_body = cv2.imdecode(np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint8), -1)
        cv_visual_hand = cv2.imdecode(np.frombuffer(image_responses[1].shot.image.data, dtype=np.uint8), -1)

        filename_body = "body_frontleft_" + str(counter) + ".jpg"
        cv2.imwrite(filename_body, cv_color_image_body)

        filename_hand = "hand_" + str(counter) + ".jpg"
        cv2.imwrite(filename_hand, cv_color_image_hand)
        time.sleep(5)
        counter += 1
    return True

    return True


if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)
