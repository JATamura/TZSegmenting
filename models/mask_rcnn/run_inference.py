# run_inference contains methods for post-prediction processing.

import cv2
import os
import supervision as sv
import torch
from detectron2.config import get_cfg
from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
from torchvision.ops import nms
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from models.mask_rcnn.configure_parameters import register_seeds


def mask_nms(masks, scores, nms_threshold=0.5):
    """
    Runs class agnostic NMS on masks/segmentations instead of the bounding boxes.
    :param      masks: (list float) List of coordinates that make up the mask output from the model.
    :param      scores: (list float) List of corresponding confidence scores given to each mask.
    :param      nms_threshold: (float) Threshold to apply mask-based class agnostic NMS.
    :return:    masks_kept (list float) -- List of masks kept after applying NMS.
    """

    # Convert all masks to Polygons
    polygons = []
    for mask in masks:
        contour = sv.mask_to_polygons(mask)
        if len(contour) > 0:
            polygons.append(Polygon(contour[0]))
        else:
            polygons.append(Polygon([]))

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    masks_kept = []
    while order:
        i = order.pop(0)
        masks_kept.append(i)
        for j in order:
            # Calculate the IoU between each polygon
            intersection = polygons[i].intersection(polygons[j]).area
            union = polygons[i].union(polygons[j]).area
            iou = intersection / union

            # Remove masks with IoU greater than the threshold
            if iou > nms_threshold:
                order.remove(j)
    return masks_kept

def apply_nms(prediction, cls_agnostic_nms=0.5, mask=False):
    nms_indices = nms(prediction["instances"].pred_boxes.tensor,
                          prediction["instances"].scores, cls_agnostic_nms)
    if mask:
        nms_indices = mask_nms(prediction["instances"].pred_masks.numpy(),
                               prediction["instances"]._fields["scores"], cls_agnostic_nms)

    pred = {"instances": Instances(image_size=prediction["instances"].image_size,
                                   pred_boxes=prediction["instances"].pred_boxes[nms_indices],
                                   scores=prediction["instances"].scores[nms_indices],
                                   pred_classes=prediction["instances"].pred_classes[nms_indices]+1,
                                   pred_masks=prediction["instances"].pred_masks[nms_indices])}

    return pred

def display_predictions(pred, img, img_name="", mask=True, alpha=0.5, output_dir=None):
    fig, ax = plt.subplots()
    v = Visualizer(img[:, :, ::-1], {"thing_classes": ['Viable', 'Non-Viable', 'Empty'],
                          "thing_colors": [(255, 0, 0), (255, 255, 0), (0, 0, 0)]},
                   scale=1.2, instance_mode=ColorMode.SEGMENTATION, font_size_scale=1.5)
    colors = []
    for label in pred["instances"].pred_classes:
        if label == 1:
            colors.append([1,0,0])
        elif label == 2:
            colors.append([1,1,0])
        elif label == 3:
            colors.append([0,0,0])
        else:
            colors.append([0,0,1])
    if mask:
        out = v.overlay_instances(
            boxes = pred["instances"].pred_boxes.to("cpu"),
            masks = pred["instances"].pred_masks.to("cpu"),
            assigned_colors = colors,
            alpha = alpha
        )
    else:
        out = v.overlay_instances(
            boxes=pred["instances"].pred_boxes.to("cpu"),
            assigned_colors=colors,
            alpha=alpha
        )
    ax.imshow(cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.axis('off')
    if output_dir:
        fig.savefig(os.path.join(output_dir, img_name), dpi=800)
    plt.show()

if __name__ == "__main__":
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
        "cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST": 0.05,
        "cfg.TEST.DETECTIONS_PER_IMAGE": 800,
        "cfg.INPUT.MIN_SIZE_TEST": 2000,
        "cfg.INPUT.MAX_SIZE_TEST": 2000,
        "cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS": False
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

    img = cv2.imread("../../datasets/dataset1/all_images/384.jpg")

    alpha = 0.0

    prediction = predictor(img)
    # print(prediction)
    display_predictions(prediction, img=img, alpha=alpha)

    pred = apply_nms(prediction, cls_agnostic_nms=0.7, mask=False)
    # print(pred)
    display_predictions(pred, img=img, alpha=alpha)

    pred = apply_nms(prediction, cls_agnostic_nms=0.7, mask=True)
    # print(pred)
    display_predictions(pred, img=img, alpha=alpha)