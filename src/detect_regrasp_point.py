"""

This script shows how to detect the grasp point in an image using the segmentation model trained with roboflow.
The basic algorithm is:
1. Load the image and the predictions from the model
2. Find candidate points, which are points on the rope that are near but not inside an obstacle
3. Find the candidate point that is closest to the center of the image

Within a larger script, you might use it like this:

```python
    from detect_regrasp_point import init_roboflow, detect_regrasp_point

    # initialize once at the top
    model = init_roboflow()

    # Then call this function for each image.
    # You need to save the image to a file first, in this case "latest_hand_img.jpg"
    test_image_filename = "latest_hand_img.jpg"
    detection = detect_regrasp_point(model, test_image_filename)
```

"""
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from roboflow import Roboflow

OBSTACLE_NEAR_PX = 10

MIN_CONFIDENCE = 0.25
OBSTACLE_INFLATION_PX = 50


class DetectionError(Exception):
    pass


@dataclass
class DetectionResult:
    grasp_px: np.ndarray
    candidates_pxs: np.ndarray
    predictions: List[Dict[str, np.ndarray]]
    img: Image.Image


def main():
    model = init_roboflow()

    test_image_filename = "test3.jpg"
    detection = detect_regrasp_point(model, test_image_filename)

    plt.figure()
    plt.imshow(detection.img)
    rng = np.random.RandomState(0)
    for pred in detection.predictions:
        points = pred["points"]
        x = [p['x'] for p in points]
        y = [p['y'] for p in points]
        plt.plot(x, y, c=cm.hsv(rng.uniform()), linewidth=2, zorder=1)
    plt.scatter(detection.candidates_pxs[:, 0], detection.candidates_pxs[:, 1], color="y", marker="x", s=100,
                label='candidates',
                zorder=2)
    plt.scatter(detection.grasp_px[0], detection.grasp_px[1], color="green", marker="o", s=100, label='grasp point',
                zorder=3)
    plt.legend()
    plt.show()

    print(grasp_px)


def detect_regrasp_point(model, test_image_filename):
    img = Image.open(test_image_filename)
    predictions = get_or_load_predictions(model, test_image_filename)
    predictions = predictions["predictions"]
    rope_class_name = "blue_rope"
    obstacle_class_name = "battery"
    # create masks based on the polygons for each class
    rope_masks = []
    obstacle_masks = []
    rope_polygons = []
    obstacle_polygons = []
    for pred in predictions:
        if pred['confidence'] < MIN_CONFIDENCE:
            continue
        class_name = pred["class"]
        points = pred["points"]
        points = np.array([(p['x'], p['y']) for p in points], dtype=int)
        mask = np.zeros([img.height, img.width], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        if class_name == rope_class_name:
            rope_polygons.append(points)
            rope_masks.append(mask)
        elif class_name == obstacle_class_name:
            obstacle_polygons.append(points)
            obstacle_masks.append(mask)
    if len(obstacle_masks) == 0:
        raise DetectionError("No obstacles detected")
    if len(rope_masks) == 0:
        raise DetectionError("No rope detected")
    # # combine the masks for each class into a single mask
    # rope_mask = combine_masks(img, rope_masks)
    # obstacle_mask = combine_masks(img, obstacle_masks)
    # # inflate the obstacle mask so that the detected grasp point isn't so close to the object that we can't grasp it
    # obstacle_mask = cv2.dilate(obstacle_mask, np.ones((50, 50), np.uint8))
    candidates_pxs = []
    for rope_poly in rope_polygons:
        for rope_p in rope_poly:
            for obstacle_poly in obstacle_polygons:
                dist = cv2.pointPolygonTest(obstacle_poly, rope_p.tolist(), True)
                # if the point is inside the obstacle, we can't grasp it
                if abs(dist + OBSTACLE_INFLATION_PX) < OBSTACLE_NEAR_PX:
                    candidates_pxs.append(rope_p)
    candidates_pxs = np.array(candidates_pxs)
    # compute the distance to the camera, approximated by the distance to the center of th image
    # for each candidate point.
    d_to_center = np.linalg.norm(candidates_pxs - np.array([img.width / 2, img.height / 2]), axis=-1)
    grasp_px = candidates_pxs[np.argmin(d_to_center)]

    return DetectionResult(grasp_px, candidates_pxs, predictions, img)


def init_roboflow():
    rf = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY'])
    project = rf.workspace().project("robotgardening")
    model = project.version(6).model
    return model


def combine_masks(img, masks):
    combined_mask = np.zeros([img.height, img.width], dtype=np.uint8)
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    return combined_mask


def get_or_load_predictions(model, test_image_filename):
    image_path = Path(test_image_filename)
    predictions_path = Path(f"{image_path.stem}_pred.json")
    if predictions_path.exists():
        with predictions_path.open("r") as f:
            predictions = json.load(f)
    else:
        model.predict(test_image_filename).save(f"{image_path.stem}_pred{image_path.suffix}")
        predictions = model.predict(test_image_filename).json()

        with predictions_path.open("w") as f:
            json.dump(predictions, f)

    return predictions


if __name__ == "__main__":
    main()
