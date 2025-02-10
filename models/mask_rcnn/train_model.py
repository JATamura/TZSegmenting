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

# calculate mean and std of all colour channels in the dataset1
def mean_and_std(path):
    channel_sums = [0, 0, 0]
    channel_squared_sums = [0, 0, 0]

    for img_name in os.listdir(path):
        img = cv2.imread(path + img_name)
        if img is not None:
            for i in range(3):
                channel_sums[i] += np.mean(img[:, :, i])
                channel_squared_sums[i] += np.mean(np.power(img[:, :, i], [2]))

    mean = [s / len(os.listdir(path)[:-1]) for s in channel_sums]
    std = [np.power((channel_squared_sums[s] / len(os.listdir(path)[:-1]) - mean[s] ** 2), [0.5])[0]
           for s in range(3)]

    return mean, std

def build_config():
    # set up the logger
    setup_logger()

    dataset_path = "../../datasets/coco/all_data/"

    # register the training and validation datasets into COCO
    register_coco_instances('train', {"thing_classes": ['Seed', 'Viable', 'Non-Viable', 'Empty'],
                                      "thing_colors": [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]},
                            dataset_path + 'images/train/annotations_train.json', dataset_path + 'images/train/')
    register_coco_instances('val', {"thing_classes": ['Seed', 'Viable', 'Non-Viable', 'Empty'],
                                    "thing_colors": [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]},
                            dataset_path + 'images/val/annotations_val.json', dataset_path + 'images/val/')
    register_coco_instances('test', {"thing_classes": ['Seed', 'Viable', 'Non-Viable', 'Empty'],
                                     "thing_colors": [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]},
                            dataset_path + 'images/test/annotations_test.json', dataset_path + 'images/test/')
    # initialise custom config
    cfg = get_cfg()
    # Mask RCNN with ResNet101
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

    # Mask RCNN with ResNeXt101
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))

    # Cascade Mask R-CNN
    # cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"))

    # set number of classes and datasets
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)

    # cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    # dataset = get_detection_dataset_dicts(
    #     cfg.DATASETS.TRAIN,
    #     filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
    #     min_keypoints=(
    #         cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE if cfg.MODEL.KEYPOINT_ON else 0
    #     ),
    #     proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    # )
    # cfg.DATALOADER.REPEAT_THRESHOLD = 0.9
    # cfg.DATALOADER.REPEAT_SQRT = True
    # cfg.DATASETS.TRAIN_REPEAT_FACTOR = [
    #     (cfg.DATASETS.TRAIN[0], RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
    #         dataset, cfg.DATALOADER.REPEAT_THRESHOLD, sqrt=cfg.DATALOADER.REPEAT_SQRT
    #     ).tolist())]

    # set custom means and stds
    mean, std = mean_and_std(dataset_path + "images/train/")
    cfg.MODEL.PIXEL_MEAN = np.array(mean, dtype=float).tolist()
    cfg.MODEL.PIXEL_STD = np.array(std, dtype=float).tolist()

    cfg.MODEL.MAX_SIZE_TRAIN = 1200

    # cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]

    # # enable cascade mask r-cnn
    cfg.MODEL.ROI_HEADS.NAME = "CascadeROIHeads"
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True

    # # standard parameters for machine learning
    # cfg.SOLVER.REFERENCE_WORLD_SIZE = 1
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 3000
    cfg.SOLVER.WARMUP_ITERS = int(cfg.SOLVER.MAX_ITER / 6)
    cfg.SOLVER.STEPS = [int(cfg.SOLVER.MAX_ITER / 3), 2 * int(cfg.SOLVER.MAX_ITER / 3)]
    cfg.SOLVER.CHECKPOINT_PERIOD = int(cfg.SOLVER.MAX_ITER / 3)

    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]]

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512, 1024]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]]

    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]

    cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "ciou"
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "ciou"
    cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
    cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 4
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 1

    cfg.MODEL.FPN.NORM = "GN"
    cfg.MODEL.ROI_BOX_HEAD.NORM = "GN"
    cfg.MODEL.ROI_MASK_HEAD.NORM = "GN"
    cfg.MODEL.RESNETS.NORM = "GN"

    # # weight controlling effect of undetected instances
    cfg.MODEL.RPN.LOSS_WEIGHT = 1.0
    cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 1.0

    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.5, 0.7]
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.RPN.POSITIVE_FRACTION = 0.7

    # overlapping bounding box threshold for instances
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 4000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 9000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 3000
    cfg.MODEL.RPN.NMS_THRESH = 0.7

    # ratio of foreground images
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    # number of foreground + background proposals
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.7

    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
    # threshold for confidence
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    # removing overlapping bounding boxes of the same class
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

    # max number of instances per image
    cfg.TEST.DETECTIONS_PER_IMAGE = 1100
    cfg.TEST.EVAL_PERIOD = 500

    return cfg

if __name__ == "__main__":
    cfg = build_config()

    # class Counter(DatasetEvaluator):
    #
    #     def reset(self):
    #         accuracy = []
    #
    #     def process(self, inputs, outputs):
    #
    #     def evaluate(self):

    # custom trainer for validation data
    class MyTrainer(DefaultTrainer):

        # @classmethod
        # def build_train_loader(cls, cfg):
        #     aug = [
        #         T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        #         T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        #         T.RandomCrop(crop_type="relative", crop_size=(0.75, 0.75)),
        #         T.RandomBrightness(0.75, 1.25),
        #         T.RandomContrast(0.75, 1.25),
        #     ]
        #     mapper = DatasetMapper(cfg, is_train=True, augmentations=aug)
        #     return build_detection_train_loader(cfg, mapper=mapper)

        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            coco_evaluator = COCOEvaluator(dataset_name, output_dir=output_folder, max_dets_per_image=1000)
            evaluator_list = [coco_evaluator]
            return DatasetEvaluators(evaluator_list)

        @classmethod
        def build_optimizer(cls, cfg, model):
            params = get_default_optimizer_params(
                model,
                base_lr=cfg.SOLVER.BASE_LR,
                weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
                bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
                weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
            )
            return maybe_add_gradient_clipping(cfg, torch.optim.AdamW)(
                params,
                lr=cfg.SOLVER.BASE_LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )

    ## training step
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()