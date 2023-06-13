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

MIN_CONFIDENCE = 0.25


class DetectionError(Exception):
    pass


@dataclass
class DetectionResult:
    grasp_px: np.ndarray
    candidates_pxs: np.ndarray
    predictions: List[Dict[str, np.ndarray]]


def init_roboflow():
    rf = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY'])
    project = rf.workspace().project("spot-vaccuming-demo")
    model = project.version(8).model
    return model


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


def viz_detection(pil_img, detection):
    plt.figure()
    plt.imshow(pil_img)
    rng = np.random.RandomState(0)
    class_colors = {}
    for pred in detection.predictions:
        points = pred["points"]
        class_name = pred["class"]
        if class_name not in class_colors:
            class_colors[class_name] = cm.hsv(rng.uniform())
        x = [p['x'] for p in points]
        y = [p['y'] for p in points]
        c = class_colors[class_name]
        plt.plot(x, y, c=c, linewidth=2, zorder=1)
    plt.scatter(detection.candidates_pxs[:, 0], detection.candidates_pxs[:, 1], color="y", marker="x", s=100,
                label='candidates',
                zorder=2)
    plt.scatter(detection.grasp_px[0], detection.grasp_px[1], color="green", marker="o", s=100, label='grasp point',
                zorder=3)
    plt.legend()
    plt.show()


def get_polys(predictions, desired_class_name):
    polys = []
    for pred in predictions:
        if pred['confidence'] < MIN_CONFIDENCE:
            continue
        class_name = pred["class"]
        points = pred["points"]
        points = np.array([(p['x'], p['y']) for p in points], dtype=int)

        if class_name == desired_class_name:
            polys.append(points)
    return polys


def get_predictions(model, test_image_filename):
    predictions = get_or_load_predictions(model, test_image_filename)
    predictions = predictions["predictions"]
    return predictions


def detect_regrasp_point(predictions, near_px, far_px):
    hose_class_name = "vacuum_hose"
    obstacle_class_name = "battery"

    hose_polygons = get_polys(predictions, hose_class_name)
    obstacle_polygons = get_polys(predictions, obstacle_class_name)

    candidates_pxs = []
    for hose_poly in hose_polygons:
        for hose_p in hose_poly:
            min_d_to_any_obstacle = np.inf
            for obstacle_poly in obstacle_polygons:
                # dist is positive if the point is outside the polygon
                dist = -cv2.pointPolygonTest(obstacle_poly, hose_p.tolist(), True)
                if dist < min_d_to_any_obstacle:
                    min_d_to_any_obstacle = dist
            # a point is a candidate if it is within some distance of an obstacle
            if near_px < min_d_to_any_obstacle < far_px:
                candidates_pxs.append(hose_p)

    if len(candidates_pxs) == 0:
        raise DetectionError("No regrasp point candidates found")

    candidates_pxs = np.array(candidates_pxs)
    # compute the distance to the camera, approximated by the distance to the bottom-center of th image
    # for each candidate point.
    # FIXME: hard coded jank
    d_to_center = np.linalg.norm(candidates_pxs - np.array([0, 400]), axis=-1)
    grasp_px = candidates_pxs[np.argmin(d_to_center)]

    return DetectionResult(grasp_px, candidates_pxs, predictions)


def detect_object_center(predictions, class_name):
    polygons = get_polys(predictions, class_name)

    if len(polygons) == 0:
        raise DetectionError(f"No {class_name} detected")
    if len(polygons) > 1:
        print("Warning: multiple objects detected")

    poly = polygons[0]
    M = cv2.moments(poly)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    grasp_point = np.array([cx, cy])

    return DetectionResult(grasp_point, np.array([grasp_point]), predictions)


def main():
    model = init_roboflow()

    test_image_filename = "test1.png"
    predictions = get_predictions(model, test_image_filename)

    regrasp_detection = detect_regrasp_point(predictions, 10, 40)
    pil_img = Image.open(test_image_filename)
    viz_detection(pil_img, regrasp_detection)

    head_detection = detect_object_center(predictions, "vacuum_head")
    viz_detection(pil_img, head_detection)

    mess_detection = detect_object_center(predictions, "mess")
    viz_detection(pil_img, mess_detection)


if __name__ == "__main__":
    main()
