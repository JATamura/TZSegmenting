import itertools
from importlib.metadata import MetadataPathFinder

from matplotlib import pyplot as plt
import numpy as np
import os, cv2

from supervision import OverlapFilter
from collections import OrderedDict, defaultdict
from detectron2.checkpoint import Checkpointer
from detectron2.data import MetadataCatalog, Metadata, DatasetMapper, get_detection_dataset_dicts
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import DatasetEvaluators, COCOEvaluator, inference_on_dataset, DatasetEvaluator
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import Instances
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.solver.build import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.data import transforms as T
from torchvision.ops import nms
import torch
import supervision as sv
from train_model import build_config

def run_inference(cfg, predictor, img_path, img_name=None, cls_agnostic_nms=1.0, display=True, mask=True):
    print(img_path[-7:])
    im = cv2.imread(img_path)
    if im is not None:
        pred = predictor(im)
        print(torch.bincount(pred["instances"].pred_classes))
        print(len(pred["instances"].pred_classes))

        nms_indices = nms(pred["instances"]._fields["pred_boxes"].tensor,
                          pred["instances"]._fields["scores"], cls_agnostic_nms)

        pred = {'instances': Instances((2560, 2048),
                                       pred_boxes=pred["instances"].pred_boxes[nms_indices],
                                       scores=pred["instances"].scores[nms_indices],
                                       pred_classes=pred["instances"].pred_classes[nms_indices],
                                       pred_masks=pred["instances"].pred_masks[nms_indices])}
        # print(len(pred["instances"].pred_classes))

        if display:
            fig, ax = plt.subplots()
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                           scale=1.2, instance_mode=ColorMode.SEGMENTATION, font_size_scale=1.5)
            if mask:
                masks = {'instances': Instances((2048, 2560),
                                                pred_classes=pred["instances"].pred_classes,
                                                pred_masks=pred["instances"].pred_masks)}
                # out = v.draw_instance_predictions(masks["instances"].to("cpu"), jittering=False)
                colors = []
                for label in pred["instances"].pred_classes:
                    if label == 1:
                        colors.append([0,1,0])
                    elif label == 2:
                        colors.append([1,0,0])
                    elif label == 3:
                        colors.append([0,0,0])
                    else:
                        colors.append([1,1,1])
                out = v.overlay_instances(masks = masks["instances"].pred_masks.to("cpu"),
                                          assigned_colors = colors)
            else:
                out = v.draw_instance_predictions(pred["instances"].to("cpu"), jittering=False)
            ax.imshow(cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
            # fig.savefig("outputs/images/seg_" + img_name, dpi=1500)
            # fig.savefig("outputs/images/seg_noSAHI.jpg", dpi=1200)
            plt.show()

        return pred

if __name__ == "__main__":
    cfg = build_config()

    ## inference step
    ## load trained weights
    cfg.OUTPUT_DIR = "model_weights/output_part1_part2"
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)

    # # test image to run inference on
    # im1 = cv2.imread("dataset1/p_train/134.jpg") # spindly seeds (84)
    # im = cv2.imread("dataset1/test/285.jpg") # as bad as it gets around 750 seeds
    # im = cv2.imread("dataset1/default/077.jpg")  # high seed count (274)
    im = cv2.imread("../../datasets/coco/all_parts/original/images/default/053.jpg") # 23
    # im = cv2.imread("dataset1/default/054.jpg") # low seed count (12-13) test image
    # im = cv2.imread("dataset1/sahi_test/cropped_normal.jpg")
    # im = cv2.imread("dataset2/DSC_4050_tff.tif")
    # im = cv2.imread("datasets/dataset2/Splicing_Test_Grey_Paper/images/test/8 Grid Tests3.jpg")

    pred = run_inference(cfg, predictor,"../../datasets/coco/all_parts/original/images/default/053.jpg", cls_agnostic_nms=0.6, mask=True)

    # def callback(image_slice: np.ndarray) -> sv.Detections:
    #     result = predictor(image_slice)
    #     print("Done")
    #     return sv.Detections.from_detectron2(result)
    #
    #
    # #
    # slicer = sv.InferenceSlicer(callback=callback, slice_wh=(1600, 1280),
    #                             overlap_ratio_wh=(0.25, 0.25),
    #                             overlap_filter=OverlapFilter("non_max_merge"),
    #                             iou_threshold=0.6)
    #
    # detections = slicer(im)
    #
    # print(len(detections.xyxy))
    # print(detections)
    #
    # d = nms(torch.from_numpy(np.array(detections.xyxy, dtype=np.float64)),
    #         torch.from_numpy(np.array(detections.confidence, dtype=np.float64)), 0.6)
    # print(len(d))
    #
    # detections = sv.Detections(detections.xyxy[d],
    #                            detections.mask[d],
    #                            detections.confidence[d],
    #                            detections.class_id[d],
    #                            detections.tracker_id,
    #                            data={})
    #
    # fig, ax = plt.subplots()
    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
    #                scale=1.2, instance_mode=ColorMode.SEGMENTATION, font_size_scale=1.5)
    # masks = {'instances': Instances(im.shape[:2],
    #                                 pred_classes=detections.class_id,
    #                                 pred_masks=detections.mask)}
    # out = v.overlay_instances(masks=masks["instances"].pred_masks,
    #                           assigned_colors=[[1, 0, 0] for i in range(len(masks["instances"].pred_masks))])
    # # out = v.draw_instance_predictions(masks["instances"].to("cpu"), jittering=False)
    # ax.imshow(cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    # # fig.savefig(img_path + "../../output/seg_" + img_path[-7:], dpi=2000)
    # fig.savefig("outputs/images/grey_test.jpg", dpi=1200)
    #
    # Build the test dataset1

    cfg.DATASETS.TEST = ("test", )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    predictor = DefaultPredictor(cfg)

    print(predictor(im))

    test_loader = build_detection_test_loader(cfg, "test")

    # Evaluate the model
    evaluator = COCOEvaluator("test", cfg, False, max_dets_per_image=1100)
    results = inference_on_dataset(predictor.model, test_loader, evaluator)
    print(results)

    # test_path = dataset_path + "test/"
    # for image in os.listdir(test_path):
    #     run_inference(test_path + image, image, True, True)