import cv2
import numpy as np
import supervision as sv
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from supervision import OverlapFilter
from torchvision.ops import nms
from model_training_and_evaluation_methods.mask_rcnn.configure_parameters import register_seeds


def main():
    image_path = "../../datasets/dataset1/all_images"
    train = "../../datasets/dataset1/coco_format/post_quality_check/model_data/train.json"
    train_and_val = "../../datasets/dataset1/coco_format/post_quality_check/model_data/train_and_val.json"
    test = "../../datasets/dataset1/coco_format/post_quality_check/model_data/test.json"
    val = "../../datasets/dataset1/coco_format/post_quality_check/model_data/val.json"

    # register the training, validation, and test datasets into COCO
    register_seeds(image_path, train_annotations=train_and_val, test_annotations=None, val_annotations=test)

    config_path = "model_weights/final_tz_segmentor/config.yaml"

    param_dict = {
        "cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST": 0.5,
        "cfg.MODEL.RPN.NMS_THRESH": 0.5,
        "cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST": 0.5,
        "cfg.TEST.DETECTIONS_PER_IMAGE": 800,
        "cfg.INPUT.MIN_SIZE_TEST": 2000,
        "cfg.INPUT.MAX_SIZE_TEST": 2000,
    }
    print("Getting model weights")

    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    for param, value in param_dict.items():
        exec(param + " = " + str(value))
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    else:
        cfg.MODEL.DEVICE = "cuda"

    predictor = DefaultPredictor(cfg)
    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = predictor(image_slice)
        print("Done")
        return sv.Detections.from_detectron2(result)

    im = cv2.imread("../../datasets/dataset2/nothing/images/DSC_4526.JPG")
    slicer = sv.InferenceSlicer(callback=callback, slice_wh=(1200, 960),
                                overlap_ratio_wh=(0.2, 0.2),
                                overlap_filter=OverlapFilter("non_max_merge"),
                                iou_threshold=0.7)

    detections = slicer(im)

    print(len(detections.xyxy))
    print(detections)

    d = nms(torch.from_numpy(np.array(detections.xyxy, dtype=np.float64)),
            torch.from_numpy(np.array(detections.confidence, dtype=np.float64)), 0.6)
    print(len(d))

    detections = sv.Detections(detections.xyxy[d],
                               detections.mask[d],
                               detections.confidence[d],
                               detections.class_id[d],
                               detections.tracker_id,
                               data={})