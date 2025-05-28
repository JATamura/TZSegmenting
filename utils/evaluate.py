import json
import cv2
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
from utils.coco import extract_annotations

def format_gt(annotations_path, image_id, image_path=None, verbose=False):
    # Loads ground truth annotations and converts them into a list of polygons and corresponding classes.

    with open(annotations_path, 'r') as file:
        data = json.load(file)
    gt_annotations = extract_annotations(data["annotations"], image_id)

    gt_classes = []
    gt_polygons = []
    for annotation in gt_annotations:
        gt_classes.append(annotation["category_id"])

        segmentation = np.array(annotation["segmentation"][0])
        segmentation = np.array(segmentation).reshape(int(len(segmentation) / 2), 2)

        polygon = Polygon(segmentation)
        gt_polygons.append(polygon)

    for idx, cls in enumerate(gt_classes):
        if cls == 1:
            gt_classes[idx] = "viable"
        elif cls == 2:
            gt_classes[idx] = "non-viable"
        elif cls == 3:
            gt_classes[idx] = "empty"
        else:
            print(cls)

    if verbose:
        fig, ax = plt.subplots()
        for idx, polygon in enumerate(gt_polygons):
            if gt_classes[idx] == "viable":
                ax.plot(*polygon.exterior.xy, c='green', linewidth=1.3)
            elif gt_classes[idx] == "non-viable":
                ax.plot(*polygon.exterior.xy, c='red', linewidth=1.3)
            elif gt_classes[idx] == "empty":
                ax.plot(*polygon.exterior.xy, c='black', linewidth=1.3)
        if image_path is not None:
            plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        # plt.title("Ground Truth Segmentation")
        plt.xticks([])
        plt.yticks([])
        plt.savefig('human.png', dpi=2000)
        plt.show()

    return gt_polygons, gt_classes

def evaluate_segmentations(model_polygons, gt_polygons, model_classes, gt_classes, model_scores,
                           iou_threshold=0.5, confidence_threshold=0.05, cls_agnostic=False):
    # Calculates the IoU between the model segmentations and ground truth segmentations (in the format of polygons).

    matches = {}
    for gt_idx, gt_polygon in enumerate(gt_polygons):
        matches[gt_idx] = {}
        matches[gt_idx]["max_iou"] = -1
        matches[gt_idx]["max_match"] = -1
        for m_idx, m_polygon in enumerate(model_polygons):

            iou = -1
            if gt_polygon.is_valid and m_polygon.is_valid:
                intersection = gt_polygon.intersection(m_polygon).area
                union = gt_polygon.union(m_polygon).area
                iou = intersection / union

            # intersection = m_polygon * gt_polygon
            # union = m_polygon + gt_polygon
            # iou = intersection.sum() / union.sum()

            if iou > matches[gt_idx]["max_iou"] and iou > iou_threshold:
                matches[gt_idx]["max_iou"] = iou
                matches[gt_idx]["max_match"] = m_idx
    for i1, v1 in matches.items():
        for i2, v2 in matches.items():
            if i1 != i2 and v1["max_match"] == v2["max_match"]:
                if v1["max_iou"] > v2["max_iou"]:
                    v2["max_iou"] = -1
                    v2["max_match"] = -1
                else:
                    v1["max_iou"] = -1
                    v1["max_match"] = -1
    if cls_agnostic:
        metrics = {"TP": len([(i, v) for i, v in matches.items()
                              if v["max_match"] >= 0 and
                              model_scores[v["max_match"]] > confidence_threshold]),
                   "FP": 0, "TN": 0, "FN": 0}
        metrics["FP"] = sum(np.array(model_scores) > confidence_threshold) - metrics["TP"]
        metrics["FN"] = len(gt_classes) - metrics["TP"]
        if (metrics["TP"] + metrics["FP"]) <= 0:
            metrics["precision"] = -1
        else:
            metrics["precision"] = metrics["TP"] / (metrics["TP"] + metrics["FP"])
        if (metrics["TP"] + metrics["FN"]) <= 0:
            metrics["recall"] = -1
        else:
            metrics["recall"] = metrics["TP"] / (metrics["TP"] + metrics["FN"])
        return metrics
    else:
        metrics = {"viable": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
                   "non-viable": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
                   "empty": {"TP": 0, "FP": 0, "TN": 0, "FN": 0}}

        for cls, metric in metrics.items():
            metric["TP"] = len([(i, v) for i, v in matches.items()
                                if v["max_match"] >= 0 and
                                gt_classes[i] == cls and
                                gt_classes[i] == model_classes[v["max_match"]] and
                                model_scores[v["max_match"]] > confidence_threshold])
            metric["FP"] = len([i for i in model_classes if i == cls]) - metric["TP"]
            metric["FN"] = len([i for i in gt_classes if i == cls]) - metric["TP"]
            if (metric["TP"] + metric["FP"]) == 0:
                metric["precision"] = 0
            else:
                metric["precision"] = metric["TP"] / (metric["TP"] + metric["FP"])

            if (metric["TP"] + metric["FN"]) == 0:
                metric["recall"] = 0
            else:
                metric["recall"] = metric["TP"] / (metric["TP"] + metric["FN"])

        return metrics
