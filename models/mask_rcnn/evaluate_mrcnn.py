import sys

from detectron2.evaluation.coco_evaluation import instances_to_coco_json

sys.path.append("/mnt/shared/scratch/jtamura/BloombergOrchidProject")

import cv2
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from mrcnn_model import build_config
from model_weights.postqc_all_parts_4_classes.model_architecture import build_config
from run_inference import inference, display_predictions
from shapely.geometry import Polygon
import supervision as sv
import numpy as np
from utils.evaluate import format_gt, evaluate_segmentations
import os
import json

def format_coco_predictions(predictor, image_path, cls_agnostic=1.0, verbose=False):
    predictions = inference(predictor, image_path, cls_agnostic)

    model_classes = predictions["instances"].pred_classes.to("cpu").tolist()
    model_scores = predictions["instances"].scores.to("cpu").tolist()
    model_segmentations = np.array(predictions["instances"].pred_masks.to("cpu").tolist())

    model_polygons = []
    for segmentation in model_segmentations:
        contour = sv.mask_to_polygons(segmentation)
        if len(contour) > 0:
            model_polygons.append(Polygon(contour[0]))
        else:
            model_polygons.append(Polygon([]))

    for idx, cls in enumerate(model_classes):
        if cls == 1:
            model_classes[idx] = "viable"
        elif cls == 2:
            model_classes[idx] = "non-viable"
        elif cls == 3:
            model_classes[idx] = "empty"
        else:
            print(cls)

    if verbose:
        fig, ax = plt.subplots()
        plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        for idx, polygon in enumerate(model_polygons):
            if model_classes[idx] == "viable":
                ax.plot(*polygon.exterior.xy, c='red', linewidth=1.3)
            elif model_classes[idx] == "non-viable":
                ax.plot(*polygon.exterior.xy, c='yellow', linewidth=1.3)
            elif model_classes[idx] == "empty":
                ax.plot(*polygon.exterior.xy, c='black', linewidth=1.3)
        # plt.title("Mask-RCNN Segmentation")
        plt.xticks([])
        plt.yticks([])
        plt.savefig('model.png', dpi=2000)
        plt.show()

    return model_polygons, model_classes, model_scores

def in_built_eval(cfg, ids, cls_agn_nms=0.5):
    predictor = DefaultPredictor(cfg)

    # Evaluate the model
    evaluator = COCOEvaluator("orchid_val", cfg, False, max_dets_per_image=800)
    evaluator._coco_api.dataset["images"] = [i for i in evaluator._coco_api.dataset["images"]
                                             if i["id"] in ids]
    evaluator._coco_api.dataset["annotations"] = [i for i in evaluator._coco_api.dataset["annotations"]
                                                  if i["image_id"] in ids]
    evaluator._do_evaluation = "annotations" in evaluator._coco_api.dataset
    evaluator.reset()

    for img in evaluator._coco_api.dataset["images"]:
        print(img["file_name"])
        coco_predictions = inference(predictor,
                                     os.path.join(image_path, img["file_name"]),
                                     cls_agn_nms)
        print(coco_predictions)
        prediction = {"image_id": img["id"],
                      "instances": instances_to_coco_json(coco_predictions["instances"], img["id"])}

        evaluator._predictions.append(prediction)
    print(evaluator._predictions)
    results = evaluator.evaluate(img_ids=ids)

    # test_loader = build_detection_test_loader(cfg, "orchid_val")
    # results = inference_on_dataset(predictor.model, test_loader, evaluator)
    print(results)
    return results

if __name__ == "__main__":
    image_path = "../../datasets/dataset1/all_images"
    train = "../../datasets/dataset1/coco/postqc_model_data/train.json"
    test = "../../datasets/dataset1/coco/postqc_model_data/test.json"
    val = "../../datasets/dataset1/coco/postqc_model_data/val.json"
    annotations_path = "../../datasets/dataset1/coco/post_quality_check/all_postqc.json"
    path_to_weights = "model_weights/postqc_all_parts_4_classes/model_final.pth"

    confidence_thresholds = [i/100 for i in range(5, 100, 5)]

    param_dict = {"cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST": confidence_thresholds[0],
                  "cfg.INPUT.MIN_SIZE_TEST": 800}

    cfg = build_config(image_path, train, test, val, param_dict)
    cfg.MODEL.WEIGHTS = path_to_weights
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)

    with open(val, 'r') as file:
        val_data = json.load(file)
    ids = [i["id"] for i in val_data["images"]]

    ## Evaluation using detectron2 evaluation (deconstructed to implement class agnostic nms if needed)
    in_built_eval(cfg, ids, 0.5)
    exit()

    ## Evaluation using my implementation
    iou_threshold = 0.5
    with open(val, 'r') as file:
        val_data = json.load(file)

    for img in os.listdir("../../datasets/dataset1/BRIN_test_2"):
        model_polygons, model_classes, model_scores = format_coco_predictions(predictor,
                                                                          "../../datasets/dataset1/BRIN_test_2/" + img,
                                                                          0.5, verbose=True)
    exit()

    for cls_iou_thresh in [0.5, 1.0]:
        for img in val_data["images"]:
            print(img["file_name"] + " mAP50:")

            model_polygons, model_classes, model_scores = format_coco_predictions(predictor,
                                                                                  os.path.join(image_path, img["file_name"]),
                                                                                  cls_iou_thresh, verbose=False)

            gt_polygons, gt_classes = format_gt(annotations_path, img["id"],
                                                os.path.join(image_path, img["file_name"]),
                                                verbose=False)

            ct_precisions = {}
            for ct in confidence_thresholds:
                print(ct)
                ct_precisions[ct] = {"cls_agnostic": {"precision": -1, "recall": -1},
                                     "viable": {"precision": -1, "recall": -1},
                                     "non-viable": {"precision": -1, "recall": -1},
                                     "empty": {"precision": -1, "recall": -1}}
                cls_agnostic_metrics = evaluate_segmentations(model_polygons, gt_polygons, model_classes, gt_classes,
                                                              model_scores,
                                                              iou_threshold=iou_threshold, confidence_threshold=ct,
                                                              cls_agnostic=True)
                if cls_agnostic_metrics["precision"] >= 0:
                    ct_precisions[ct]["cls_agnostic"]["precision"] = cls_agnostic_metrics["precision"]
                if cls_agnostic_metrics["recall"] >= 0:
                    ct_precisions[ct]["cls_agnostic"]["recall"] = cls_agnostic_metrics["recall"]

                cls_metrics = evaluate_segmentations(model_polygons, gt_polygons, model_classes, gt_classes, model_scores,
                                                     iou_threshold=iou_threshold, confidence_threshold=ct)
                for k, v in cls_metrics.items():
                    if cls_metrics[k]["precision"] >= 0:
                        ct_precisions[ct][k]["precision"] = cls_metrics[k]["precision"]
                    if cls_metrics[k]["recall"] >= 0:
                        ct_precisions[ct][k]["recall"] = cls_metrics[k]["recall"]
                print(ct_precisions[ct])

            img_precisions[img["id"]] = ct_precisions

        mean_average_precision = {"cls_agnostic": {"precision": -1, "recall": -1},
                                  "viable": {"precision": -1, "recall": -1},
                                  "non-viable": {"precision": -1, "recall": -1},
                                  "empty": {"precision": -1, "recall": -1}}

        for img, ct_precisions in img_precisions.items():
            for ct, precisions in ct_precisions.items():
                for cls, metrics in precisions.items():
                    if metrics["precision"] >= 0:
                        if mean_average_precision[cls]["precision"] < 0:
                            mean_average_precision[cls]["precision"] = 0
                        mean_average_precision[cls]["precision"] += metrics["precision"]
                    if metrics["recall"] >= 0:
                        if mean_average_precision[cls]["recall"] < 0:
                            mean_average_precision[cls]["recall"] = 0
                        mean_average_precision[cls]["recall"] += metrics["recall"]

        for k, v in mean_average_precision.items():
            if mean_average_precision[k]["precision"] > 0:
                mean_average_precision[k]["precision"] /= (len(val_data["images"]) * len(confidence_thresholds))
            if mean_average_precision[k]["recall"] > 0:
                mean_average_precision[k]["recall"] /= (len(val_data["images"]) * len(confidence_thresholds))

        print("mAP" + str(iou_threshold))
        print(mean_average_precision)