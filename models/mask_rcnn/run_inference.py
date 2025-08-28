# run_inference contains methods for post-prediction processing.

import importlib
import cv2
import os
import numpy as np
import supervision as sv
import torch
from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
from supervision import OverlapFilter
from torchvision.ops import nms
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer, ColorMode

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
    if mask:
        nms_indices = mask_nms(prediction["instances"].pred_masks.numpy(),
                               prediction["instances"]._fields["scores"], cls_agnostic_nms)
    else:
        nms_indices = nms(prediction["instances"].pred_boxes.tensor,
                          prediction["instances"].scores, cls_agnostic_nms)

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
            colors.append([0,1,0])
        elif label == 2:
            colors.append([1,0,0])
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
    dataset_path = "../../datasets/dataset1/all_images"
    train = "../../datasets/dataset1/coco/postqc_model_data/train.json"
    test = "../../datasets/dataset1/coco/postqc_model_data/test.json"
    val = "../../datasets/dataset1/coco/postqc_model_data/val.json"

    model_path = "output_07_29_cascade_4"
    model_architecture = importlib.import_module("model_weights." + model_path + ".model_architecture")

    param_dict = {
        "cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST": 0.5,
        "cfg.MODEL.RPN.NMS_THRESH": 0.5,
        "cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST": 0.05,
        "cfg.TEST.DETECTIONS_PER_IMAGE": 800,
        "cfg.INPUT.MIN_SIZE_TEST": 1200,
        "cfg.INPUT.MAX_SIZE_TEST": 1200,
    }
    cfg = model_architecture.build_config(dataset_path, train, test, val, param_dict)

    ## load trained weights
    cfg.OUTPUT_DIR = os.path.join("model_weights", model_path)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)

    # pred = inference(predictor, "../../datasets/dataset1/BRIN_test_2/00 VANDOPSIS LISSOCHILOIDES F-2023 TZ 30aprl25 n.jpg",
    #                  cls_agnostic_nms=0.5, mask=False)
    # display_predictions(cfg, pred, im=cv2.imread("../../datasets/dataset1/BRIN_test_2/00 VANDOPSIS LISSOCHILOIDES F-2023 TZ 30aprl25 n.jpg"),
    #                     img_name="box")
    img = cv2.imread("../../datasets/dataset1/all_images/381.jpg")
    prediction = predictor(img)
    # print(prediction)
    display_predictions(prediction, img=img,
                        img_name="053")
    pred = apply_nms(prediction, cls_agnostic_nms=0.7, mask=False)
    # print(pred)
    display_predictions(pred, img=img,
                        img_name="053_box")
    pred = apply_nms(pred, cls_agnostic_nms=0.7, mask=True)
    # print(pred)
    display_predictions(pred, img=img,
                        img_name="053_box_mask")


    # for img in os.listdir("../../datasets/dataset1/BRIN_test_2"):
    #     pred = inference(predictor, "../../datasets/dataset1/BRIN_test_2/" + img,
    #                      cls_agnostic_nms=0.6)
    #     display_predictions(cfg, pred, im=cv2.imread("../../datasets/dataset1/BRIN_test_2/" + img),
    #                         img_name=img)
    exit()

    # with open(val, "r") as file:
    #     val_data = json.load(file)
    # for img in [val_data["images"][80]]:
    #
    #     pred = inference(predictor, os.path.join(dataset_path, img["file_name"]),
    #                          cls_agnostic_nms=0.3)
    #     display_predictions(cfg, pred, im=cv2.imread(os.path.join(dataset_path, img["file_name"])))
    # exit()

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