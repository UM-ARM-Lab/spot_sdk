import itertools
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
from sklearn.cluster import KMeans

MIN_CONFIDENCE = 0.25
MODEL_VERSION = 12


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
    model = project.version(MODEL_VERSION).model
    return model


def get_or_load_predictions(model, test_image_filename):
    image_path = Path(test_image_filename)
    predictions_path = Path(f"{image_path.stem}_pred.json")
    if predictions_path.exists():
        with predictions_path.open("r") as f:
            predictions = json.load(f)
    else:
        predictions = model.predict(test_image_filename).json()

        with predictions_path.open("w") as f:
            json.dump(predictions, f)

    return predictions


def viz_detection(rgb_np, detection):
    plt.figure()
    plt.imshow(rgb_np)
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


def get_points_within_dist(far_px, input_pxs, near_px, obstacle_polygons):
    candidates_pxs = []
    for hose_p in input_pxs:
        min_d_to_any_obstacle = min_dist_to_obstacles(obstacle_polygons, hose_p)
        # a point is a candidate if it is within some distance of an obstacle
        if near_px < min_d_to_any_obstacle < far_px:
            candidates_pxs.append(hose_p)
    if len(candidates_pxs) == 0:
        raise DetectionError("No regrasp point candidates found")

    candidates_pxs = np.array(candidates_pxs)
    return candidates_pxs


def detect_regrasp_point_old(predictions, near_px, far_px):
    hose_class_name = "vacuum_hose"
    obstacle_class_name = "battery"

    hose_polygons = get_polys(predictions, hose_class_name)
    obstacle_polygons = get_polys(predictions, obstacle_class_name)

    input_pxs = np.concatenate(hose_polygons, axis=0)
    candidates_pxs = get_points_within_dist(far_px, input_pxs, near_px, obstacle_polygons)

    # compute the distance to the camera, approximated by the distance to the bottom-center of th image
    # for each candidate point.
    # FIXME: hard coded jank
    d_to_center = np.linalg.norm(candidates_pxs - np.array([320, 400]), axis=-1)
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

    detection = DetectionResult(grasp_point, np.array([grasp_point]), predictions)

    return detection


def fit_hose_model(hose_polygons):
    hose_points = np.concatenate(hose_polygons, 0)
    kmeans = KMeans(n_clusters=6, random_state=0, n_init="auto").fit(hose_points)
    clusters = kmeans.cluster_centers_

    # organize the points into line segments with the shortest total length
    def _len_cost(points):
        deltas = points[1:] - points[:-1]
        lengths = np.linalg.norm(deltas, axis=-1)
        return lengths.sum()

    min_length = np.inf
    best_ordered_hose_points = None
    for permutation in itertools.permutations(range(clusters.shape[0])):
        ordered_hose_points = clusters[list(permutation)]
        cost = _len_cost(ordered_hose_points)
        if cost < min_length:
            min_length = cost
            best_ordered_hose_points = ordered_hose_points

    return best_ordered_hose_points


def detect_regrasp_point(predictions, ideal_dist_to_obs):
    hose_class_name = "vacuum_hose"
    obstacle_class_name = "battery"

    hose_polygons = get_polys(predictions, hose_class_name)
    obstacle_polygons = get_polys(predictions, obstacle_class_name)

    ordered_hose_points = fit_hose_model(hose_polygons)
    n = ordered_hose_points.shape[0]

    # Find the angle of each segment in the hose with respect to the X axis, between -pi/2 and pi/2.
    # Pick the point which minimizes the following cost
    # alpha * angle of hose at point + (1 - alpha) * abs(dist_to_obstacle - ideal_dist_to_obstacle)

    alpha = 0.5
    deltas = ordered_hose_points[1:] - ordered_hose_points[:-1]
    angles_costs = np.zeros(n)
    flat_hose_points = []
    for i in range(n):
        if i == 0:
            delta = deltas[0]
        if i > 0:
            delta = deltas[i - 1]
        angle = min_angle_to_x_axis(delta)
        angles_costs[i] = abs(angle)

    dist_costs = np.zeros(n)
    for i, p in enumerate(ordered_hose_points):
        min_d_to_any_obstacle = min_dist_to_obstacles(obstacle_polygons, p)
        dist_costs[i] = abs(min_d_to_any_obstacle - ideal_dist_to_obs)

    total_cost = alpha * angles_costs + (1 - alpha) * dist_costs
    min_cost_idx = np.argmin(total_cost)
    best_px = ordered_hose_points[min_cost_idx]

    return DetectionResult(best_px, ordered_hose_points, predictions)


def min_dist_to_obstacles(obstacle_polygons, p):
    min_d_to_any_obstacle = np.inf
    for obstacle_poly in obstacle_polygons:
        # dist is positive if the point is outside the polygon
        dist = -cv2.pointPolygonTest(obstacle_poly, p.tolist(), True)
        if dist < min_d_to_any_obstacle:
            min_d_to_any_obstacle = dist
    return min_d_to_any_obstacle


def min_angle_to_x_axis(delta):
    angle = np.arctan2(delta[1], delta[0])
    neg_angle = np.arctan2(-delta[1], -delta[0])
    if abs(angle) < abs(neg_angle):
        return angle
    else:
        return neg_angle


def main():
    model = init_roboflow()

    test_image_filename = "above3.png"
    predictions = get_predictions(model, test_image_filename)

    detection = detect_regrasp_point(predictions, 50)

    rgb_pil = Image.open(test_image_filename)
    rgb_np = np.asarray(rgb_pil)

    viz_detection(rgb_pil, detection)


if __name__ == "__main__":
    main()
