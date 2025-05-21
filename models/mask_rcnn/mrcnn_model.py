import weakref

import cv2
import os
import numpy as np
import torch
from fvcore.common.checkpoint import Checkpointer
from torchvision.ops import nms

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import get_detection_dataset_dicts
from detectron2.data.datasets import register_coco_instances
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.engine import DefaultTrainer, hooks
from detectron2.evaluation import DatasetEvaluators, COCOEvaluator
from detectron2.modeling import build_model
from detectron2.solver.build import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.engine import launch
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.structures import Instances


# calculate mean and std of all colour channels in the dataset1
def mean_and_std(path):
    channel_sums = [0, 0, 0]
    channel_squared_sums = [0, 0, 0]

    for img_name in os.listdir(path):
        img = cv2.imread(os.path.join(path, img_name))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None:
            for i in range(3):
                channel_sums[i] += np.mean(img[:, :, i])
                channel_squared_sums[i] += np.mean(np.power(img[:, :, i], [2]))

    mean = [s / len(os.listdir(path)[:-1]) for s in channel_sums]
    std = [np.power((channel_squared_sums[s] / len(os.listdir(path)[:-1]) - mean[s] ** 2), [0.5])[0]
           for s in range(3)]

    return mean, std

def build_config(img_path, train_annotations, test_annotations, val_annotations, param_dict={}):
    # set up the logger
    setup_logger()

    # register the training and validation datasets into COCO
    register_coco_instances('orchid_train',
                            {"thing_classes": ['Viable', 'Non-Viable', 'Empty'],
                             "thing_colors": [(255, 0, 0), (0, 255, 0), (0, 0, 255)]},
                            train_annotations, img_path)
    register_coco_instances('orchid_val',
                            {"thing_classes": ['Viable', 'Non-Viable', 'Empty'],
                             "thing_colors": [(255, 0, 0), (0, 255, 0), (0, 0, 255)]},
                            val_annotations, img_path)
    register_coco_instances('orchid_test',
                            {"thing_classes": ['Viable', 'Non-Viable', 'Empty'],
                             "thing_colors": [(255, 0, 0), (0, 255, 0), (0, 0, 255)]},
                            test_annotations, img_path)
    # initialise custom config
    cfg = get_cfg()
    # Mask RCNN with ResNet101
    # config = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

    # Mask RCNN with ResNeXt101
    # config = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"

    # Cascade Mask R-CNN
    config = "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"

    cfg.merge_from_file(model_zoo.get_config_file(config))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config)

    # set number of classes and datasets
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.DATASETS.TRAIN = ("orchid_train",)
    cfg.DATASETS.TEST = ("orchid_val",)

    # cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    # dataset = get_detection_dataset_dicts(
    #     cfg.DATASETS.TRAIN,
    #     filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
    #     min_keypoints=(
    #         cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE if cfg.MODEL.KEYPOINT_ON else 0
    #     ),
    #     proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    # )
    # cfg.DATALOADER.REPEAT_THRESHOLD = 0.5
    # cfg.DATALOADER.REPEAT_SQRT = True
    # cfg.DATASETS.TRAIN_REPEAT_FACTOR = [
    #     (cfg.DATASETS.TRAIN[0], RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
    #         dataset, cfg.DATALOADER.REPEAT_THRESHOLD, sqrt=cfg.DATALOADER.REPEAT_SQRT
    #     ).tolist())]

    # set custom means and stds
    mean, std = mean_and_std(os.path.join(img_path))
    cfg.MODEL.PIXEL_MEAN = np.array(mean, dtype=float).tolist()
    cfg.MODEL.PIXEL_STD = np.array(std, dtype=float).tolist()

    cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]

    # # enable cascade mask r-cnn
    cfg.MODEL.ROI_HEADS.NAME = "CascadeROIHeads"
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True

    # # standard parameters for machine learning
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 4
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 8000
    cfg.SOLVER.WARMUP_ITERS = int(cfg.SOLVER.MAX_ITER / 5)
    # cfg.SOLVER.STEPS = [2000, 3000]
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.WEIGHT_DECAY = 0.01

    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"

    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512, 1024]]
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64], [64, 128],
                                        [128, 256], [256, 512], [512, 720]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]]

    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]

    cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "ciou"
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "ciou"
    cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
    cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 4
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 1
    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV = 8
    cfg.MODEL.BACKBONE.FREEZE_AT = 0

    cfg.MODEL.FPN.NORM = "GN"
    cfg.MODEL.ROI_BOX_HEAD.NORM = "GN"
    cfg.MODEL.ROI_MASK_HEAD.NORM = "GN"
    cfg.MODEL.RESNETS.NORM = "GN"

    # # weight controlling effect of undetected instances
    cfg.MODEL.RPN.LOSS_WEIGHT = 1.0
    cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 1.0
    # cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.5

    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.5, 0.7]
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.RPN.POSITIVE_FRACTION = 0.8

    # overlapping bounding box threshold for instances
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 10000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 6000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 7500
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 4500
    cfg.MODEL.RPN.NMS_THRESH = 0.5

    # ratio of foreground images
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    # number of foreground + background proposals
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.8

    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
    # threshold for confidence
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    # removing overlapping bounding boxes of the same class
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

    # max number of instances per image
    cfg.TEST.DETECTIONS_PER_IMAGE = 800
    cfg.TEST.EVAL_PERIOD = 500

    for param, value in param_dict.items():
        exec(param + " = " + str(value))

    return cfg

class MyEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, max_dets_per_image, cls_agnostic_nms):
        super().__init__(dataset_name, max_dets_per_image=max_dets_per_image)
        self.cls_agnostic_nms = cls_agnostic_nms

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            instances = output["instances"].to(self._cpu_device)
            nms_indices = nms(instances._fields["pred_boxes"].tensor,
                              instances._fields["scores"], self.cls_agnostic_nms)
            nms_prediction  = {'instances': Instances(image_size=instances.image_size,
                               pred_boxes=instances.pred_boxes[nms_indices],
                               scores=instances.scores[nms_indices],
                               pred_classes=instances.pred_classes[nms_indices],
                               pred_masks=instances.pred_masks[nms_indices])}
            print(nms_prediction)

            prediction["instances"] = instances_to_coco_json(nms_prediction["instances"], input["image_id"])
            self._predictions.append(prediction)

    def evaluate(self, img_ids=None):
        results = {}
        for k, v in super().evaluate().items():
            results[k+"_cls_agn_nms"] = v
        return results

# custom trainer for validation data
class MyTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        aug = [
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.ResizeShortestEdge(short_edge_length=[800, 800]),
            # T.RandomCrop("absolute", (800, 800)),
            T.RandomSaturation(0.95, 1.05),
            T.RandomBrightness(0.95, 1.05),
            T.RandomContrast(0.95, 1.05),
        ]
        mapper = DatasetMapper(cfg, is_train=True, augmentations=aug)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        coco_evaluator = COCOEvaluator(dataset_name,
                                       output_dir=output_folder,
                                       max_dets_per_image=800)
        nms_evaluator = MyEvaluator(dataset_name, 800,0.5)
        evaluator_list = [coco_evaluator, nms_evaluator]
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

def main():
    dataset_path = "../../datasets/dataset1/all_images"
    train = "../../datasets/dataset1/coco/postqc_model_data/train.json"
    test = "../../datasets/dataset1/coco/postqc_model_data/test.json"
    val = "../../datasets/dataset1/coco/postqc_model_data/val.json"

    param_dict = {}

    cfg = build_config(dataset_path, train, test, val, param_dict)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            trainer.build_model(cfg),
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(trainer),
        )
    trainer.register_hooks(
        [hooks.BestCheckpointer(eval_period=cfg.TEST.EVAL_PERIOD, checkpointer=checkpointer, val_metric="bbox/AP50")]
    )
    # trainer.train()
    return trainer.train()

if __name__ == "__main__":
    # main()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    launch(main, 4)