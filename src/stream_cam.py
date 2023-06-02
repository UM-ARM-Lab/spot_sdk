import argparse
import io
import sys

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient
from PIL import Image

import numpy as np
import cv2
from matplotlib import pyplot as plt
import colorsys

#include ""

def convertToHSV(rgb):
    hsv = np.array(rgb, copy=True)
    for i in rgb:
        hsv[i] = colorsys.rgb_to_hsv(*i)
    return hsv

def stream_cam(config, logger):
    bosdyn.client.util.setup_logging()
    sdk = bosdyn.client.create_standard_sdk('spot-stream-cam')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    image_client = robot.ensure_client(ImageClient.default_service_name)
    img_src = "hand_color_image"
    sources = image_client.list_image_sources()
    src_names = [source.name for source in sources]
    if (img_src not in src_names):
        logger.exception("Image source doesn't exist")
    
    while True:
        image_response = image_client.get_image_from_sources([img_src])[0]
        # get binary pillow image
        pil_image = Image.open(io.BytesIO(image_response.shot.image.data)).convert('RGB')
        # convert to cv mat
        cv_color_image = np.array(pil_image)
        # fix color to be BGR
        cv_color_image = cv_color_image[:, :, ::-1].copy()

        # grey
        grey = cv2.cvtColor(cv_color_image, cv2.COLOR_BGR2GRAY)
        grey = cv2.Canny(cv_color_image, 200, 300)
        blurred_grey = cv2.GaussianBlur(grey, (5,5), 0)
        trimmed_grey = grey[20:-20,20:-20]
        mask = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,51,10)
        
        # finding contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        for i, contour in enumerate(contours):
            area = int(cv2.contourArea(contour))
            if (area > 4000):
                filtered_contours.append(contour)
        cv2.drawContours(cv_color_image, filtered_contours, -1, (0, 255, 0), 2)
        # image as Mx3 array
        # cv_color_image = cv2.imread("/home/aliryckman/spot-sdk/src/Flag_of_Russia.png", cv2.IMREAD_COLOR)
        #reshape_color = cv_color_image.reshape((-1,3))
        
        #reshape_hsv = convertToHSV(reshape_color)
        # # extract just red
        #red_color = reshape_color[:,0]
        #red_color = np.float32(red_color)
        # # kmeans on the red
        #plt.hist(red_color, 256, [0,256]),plt.show()

        cv2.imshow('color', cv_color_image)
        # cv2.imshow("blurred", blurred_grey)
        cv2.imshow("mask", mask)
        cv2.waitKey(1)
        
    

def main(argv):
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)
    logger = bosdyn.client.util.get_logger()
    try:
        stream_cam(options, logger)
        return True
    except Exception as exc:
        logger.exception("Threw an exception")
        return False
    
if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)