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
    polygons = []
    for mask in masks:
        contour = sv.mask_to_polygons(mask)
        if len(contour) > 0:
            polygons.append(Polygon(contour[0]))
        else:
            polygons.append(Polygon([]))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        for j in order:
            # Calculate the IoU between the two polygons
            intersection = polygons[i].intersection(polygons[j]).area
            union = polygons[i].union(polygons[j]).area
            iou = intersection / union

            # intersection = masks[i] * masks[j]
            # union = masks[i] + masks[j]
            # iou = intersection.sum() / union.sum()

            # Remove masks with IoU greater than the threshold
            if iou > nms_threshold:
                order.remove(j)
    return keep

def apply_nms(prediction, cls_agnostic_nms=0.5, mask=False):
    if mask:
        # print("Applying mask NMS")
        nms_indices = mask_nms(prediction["instances"].pred_masks.numpy(),
                               prediction["instances"]._fields["scores"], cls_agnostic_nms)
    else:
        # print("Applying box NMS")
        nms_indices = nms(prediction["instances"].pred_boxes.tensor,
                          prediction["instances"].scores, cls_agnostic_nms)

    pred = {"instances": Instances(image_size=prediction["instances"].image_size,
                                   pred_boxes=prediction["instances"].pred_boxes[nms_indices],
                                   scores=prediction["instances"].scores[nms_indices],
                                   pred_classes=prediction["instances"].pred_classes[nms_indices]+1,
                                   pred_masks=prediction["instances"].pred_masks[nms_indices])}

    return pred

def display_predictions(cfg, pred, im, img_name="", mask=True):
    fig, ax = plt.subplots()
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=1.2, instance_mode=ColorMode.SEGMENTATION, font_size_scale=1.5)
    if mask:
        colors = []
        for label in pred["instances"].pred_classes:
            if label == 1:
                colors.append([0,1,0])
            elif label == 2:
                colors.append([1,0,0])
            elif label == 3:
                colors.append([0,0,0])
            else:
                print(label)
                colors.append([1,1,1])
        out = v.overlay_instances(masks = pred["instances"].pred_masks.to("cpu"),
                                  assigned_colors = colors)
    else:
        out = v.draw_instance_predictions(pred["instances"].to("cpu"), jittering=False)
    ax.imshow(cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.axis('off')
    fig.savefig("outputs/seg_" + img_name, dpi=800)
    plt.show()

if __name__ == "__main__":
    dataset_path = "../../datasets/dataset1/all_images"
    train = "../../datasets/dataset1/coco/postqc_model_data/train.json"
    test = "../../datasets/dataset1/coco/postqc_model_data/test.json"
    val = "../../datasets/dataset1/coco/postqc_model_data/val.json"

    model_path = "output_02_05_no_freeze"
    model_architecture = importlib.import_module("model_weights." + model_path + ".model_architecture")

    param_dict = {"cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST": 0.8,
                  "cfg.MODEL.RPN.NMS_THRESH": 0.8,
                  "cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST": 0.05,
                  "cfg.TEST.DETECTIONS_PER_IMAGE": 1000,
                  "cfg.INPUT.MIN_SIZE_TEST": 800}
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
    img = cv2.imread("../../datasets/dataset1/BRIN_test_2/00 VANDOPSIS LISSOCHILOIDES F-2023 TZ 30aprl25 n.jpg")
    prediction = predictor(img)
    pred = apply_nms(prediction, cls_agnostic_nms=0.5, mask=True)
    display_predictions(cfg, pred, im=cv2.imread("../../datasets/dataset1/BRIN_test_2/00 VANDOPSIS LISSOCHILOIDES F-2023 TZ 30aprl25 n.jpg"),
                        img_name="box_mask")


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

    fig, ax = plt.subplots()
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=1.2, instance_mode=ColorMode.SEGMENTATION, font_size_scale=1.5)
    masks = {'instances': Instances(im.shape[:2],
                                    pred_classes=detections.class_id,
                                    pred_masks=detections.mask)}
    out = v.overlay_instances(masks=masks["instances"].pred_masks,
                              assigned_colors=[[1, 0, 0] for i in range(len(masks["instances"].pred_masks))])
    # out = v.draw_instance_predictions(masks["instances"].to("cpu"), jittering=False)
    ax.imshow(cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    # # fig.savefig(img_path + "../../output/seg_" + img_path[-7:], dpi=2000)
    # fig.savefig("outputs/images/grey_test.jpg", dpi=1200)
    plt.show()