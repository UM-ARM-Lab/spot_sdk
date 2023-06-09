import json
import os
from pathlib import Path

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


def main():
    rf = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY'])
    project = rf.workspace().project("robotgardening")
    model = project.version(6).model

    test_image_filename = "test3.jpg"
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

    plt.figure()
    plt.imshow(img)
    rng = np.random.RandomState(0)
    for pred in predictions:
        points = pred["points"]
        x = [p['x'] for p in points]
        y = [p['y'] for p in points]
        plt.plot(x, y, c=cm.hsv(rng.uniform()), linewidth=2, zorder=1)
    plt.scatter(candidates_pxs[:, 0], candidates_pxs[:, 1], color="red", marker="x", s=100, label='candidates',
                zorder=2)
    plt.scatter(grasp_px[0], grasp_px[1], color="green", marker="o", s=100, label='grasp point', zorder=3)
    plt.legend()
    plt.show()

    print(grasp_px)


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
