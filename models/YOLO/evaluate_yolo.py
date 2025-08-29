import sys

sys.path.append("/mnt/shared/scratch/jtamura/BloombergOrchidProject")
sys.path.append("/C:/Users/jta10wk/PycharmProjects/BloombergOrchidProject")

import cv2
from matplotlib import pyplot as plt
import json
import os
from shapely.geometry import Polygon
from ultralytics import YOLO
from misc.evaluate import format_gt, evaluate_segmentations
from detectron2.structures import Instances, Boxes
from detectron2.evaluation.coco_evaluation import instances_to_coco_json, COCOEvaluator
from models.mask_rcnn.configure_parameters import build_config

def yolo_to_coco_evaluation(model, cls_agnostic=True):

    image_path = "../../datasets/dataset1/all_images"
    train = "../../datasets/dataset1/coco_format/post_quality_check/model_data/train.json"
    test = "../../datasets/dataset1/coco_format/post_quality_check/model_data/test.json"
    val = "../../datasets/dataset1/coco_format/post_quality_check/model_data/val.json"
    cfg = build_config(image_path, train, test, val)
    # TODO: Should the second argument here not be `tasks = 'segm'`? I'm not sure cfg is correct
    evaluator = COCOEvaluator("orchid_val", cfg, False, max_dets_per_image=800)
    evaluator.reset()

    with open(val, 'r') as file:
        val_data = json.load(file)

    for img in val_data["images"]:
        results = model(os.path.join(image_path, img["file_name"]),
                        conf=0.05, agnostic_nms=cls_agnostic, max_det=800,
                        imgsz=(img["height"], img["width"]))
        coco_predictions = Instances(results[0].orig_shape,
                                     pred_boxes=Boxes(results[0].boxes.xyxy),
                                     scores=results[0].boxes.conf,
                                     pred_classes=results[0].boxes.cls+1,
                                     pred_masks=results[0].masks.data > 0)
        prediction = {"image_id": img["id"],
                      "instances": instances_to_coco_json(coco_predictions, img["id"])}
        evaluator._predictions.append(prediction)
    results = evaluator.evaluate()
    print(results)
    return results

def format_yolo_predictions(results, cls_agnostic=True, verbose=False):

    model_polygons = []
    for mask in results[0].masks:
        model_polygons.append(Polygon(mask.xy[0]))

    model_classes = []
    for cls in results[0].boxes.cls.tolist():
        if cls == 0:
            model_classes.append("viable")
        elif cls == 1:
            model_classes.append("non-viable")
        elif cls == 2:
            model_classes.append("empty")
        else:
            print(cls)

    model_scores = results[0].boxes.conf.tolist()

    if verbose:
        fig, ax = plt.subplots()
        plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        for idx, polygon in enumerate(model_polygons):
            if model_classes[idx] == "viable":
                ax.plot(*polygon.exterior.xy, c='red')
            elif model_classes[idx] == "non-viable":
                ax.plot(*polygon.exterior.xy, c='blue')
            elif model_classes[idx] == "empty":
                ax.plot(*polygon.exterior.xy, c='black')
        plt.title("YOLOv11 Segmentation")
        plt.xticks([])
        plt.yticks([])
        plt.show()

    return model_polygons, model_classes, model_scores

def in_built_eval(model, data_path):
    results = model.val(data=data_path, max_det=800, conf=0.05, imgsz=2560, save=False)
    return results

if __name__ == "__main__":

    image_path = "../../datasets/dataset1/all_images"
    path_to_weights = "model_weights/post_qc_yolo_model_11/weights/best.pt"
    annotations_path = "../../datasets/dataset1/coco_format/post_quality_check/base_dataset.json"
    val = "../../datasets/dataset1/coco_format/post_quality_check/model_data/val.json"

    model = YOLO(path_to_weights)

    ## Evaluation using YOLO evaluation
    # in_built_eval(model, "../../datasets/dataset1/yolo/post_qc/data.yaml")
    # exit()

    ## Evaluation using detectron2 evaluation
    results = yolo_to_coco_evaluation(model, True)

    ## Evaluation using my implementation
    confidence_thresholds = [i / 100 for i in range(5, 100, 5)]
    iou_threshold = 0.5

    with open(val, 'r') as file:
        val_data = json.load(file)

    img_precisions = {}

    for img in val_data["images"]:
        print(img["file_name"] + " mAP50:")
        results = model(os.path.join(image_path, img["file_name"]),
                        conf=0.05, agnostic_nms=True, max_det=800,
                        imgsz=(img["height"], img["width"]))

        model_polygons, model_classes, model_scores = format_yolo_predictions(results,
                                                                              verbose=False)

        gt_polygons, gt_classes = format_gt(annotations_path, img["id"],
                                            os.path.join(image_path, img["file_name"]),
                                            verbose=False)

        ct_precisions = {}
        for ct in confidence_thresholds:
            print(ct)
            ct_precisions[ct] = {"cls_agnostic": {"precision": None, "recall": None},
                                 "viable": {"precision": None, "recall": None},
                                 "non-viable": {"precision": None, "recall": None},
                                 "empty": {"precision": None, "recall": None}}
            cls_agnostic_metrics = evaluate_segmentations(model_polygons, gt_polygons,
                                                          model_classes, gt_classes,
                                                          model_scores,
                                                          iou_threshold=iou_threshold,
                                                          confidence_threshold=ct,
                                                          cls_agnostic=True)

            if cls_agnostic_metrics["precision"] is not None:
                ct_precisions[ct]["cls_agnostic"]["precision"] = cls_agnostic_metrics["precision"]
            if cls_agnostic_metrics["recall"] is not None:
                ct_precisions[ct]["cls_agnostic"]["recall"] = cls_agnostic_metrics["recall"]

            cls_metrics = evaluate_segmentations(model_polygons, gt_polygons, model_classes, gt_classes, model_scores,
                                                 iou_threshold=iou_threshold, confidence_threshold=ct)
            for k, v in cls_metrics.items():
                if cls_metrics[k]["precision"] is not None:
                    ct_precisions[ct][k]["precision"] = cls_metrics[k]["precision"]
                if cls_metrics[k]["recall"] is not None:
                    ct_precisions[ct][k]["recall"] = cls_metrics[k]["recall"]
            print(ct_precisions[ct])

        img_precisions[img["id"]] = ct_precisions

    mean_average_precision = {"cls_agnostic": {"precision": None, "recall": None},
                              "viable": {"precision": None, "recall": None},
                              "non-viable": {"precision": None, "recall": None},
                              "empty": {"precision": None, "recall": None}}

    for img, ct_precisions in img_precisions.items():
        for ct, precisions in ct_precisions.items():
            for cls, metrics in precisions.items():
                if metrics["precision"] is not None:
                    if mean_average_precision[cls]["precision"] is None:
                        mean_average_precision[cls]["precision"] = 0
                    mean_average_precision[cls]["precision"] += metrics["precision"]
                if metrics["recall"] is not None:
                    if mean_average_precision[cls]["recall"] is None:
                        mean_average_precision[cls]["recall"] = 0
                    mean_average_precision[cls]["recall"] += metrics["recall"]

    for k, v in mean_average_precision.items():
        if mean_average_precision[k]["precision"] is not None:
            mean_average_precision[k]["precision"] /= (len(val_data["images"]) * len(confidence_thresholds))
        if mean_average_precision[k]["recall"] is not None:
            mean_average_precision[k]["recall"] /= (len(val_data["images"]) * len(confidence_thresholds))

    print("mAP" + str(iou_threshold))
    print(mean_average_precision)