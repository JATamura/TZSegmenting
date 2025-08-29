import datetime
import json
import cv2
import os
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger

def mean_and_std(img_path):
    """
    Calculates the mean and standard deviation of the BGR channels of the whole dataset.
    :param      img_path: (string) Path to training images.
    :return     mean: (list float) Mean BGR values for the images in the training dataset.
    :return     std: (list float) Standard deviation of the BGR values for the images in the training dataset.
    """
    channel_sums = [0, 0, 0]
    channel_squared_sums = [0, 0, 0]

    for img_name in os.listdir(img_path):
        img = cv2.imread(os.path.join(img_path, img_name))
        if img is not None:
            for i in range(3):
                channel_sums[i] += np.mean(img[:, :, i])
                channel_squared_sums[i] += np.mean(np.power(img[:, :, i], [2]))

    mean = [s / len(os.listdir(img_path)[:-1]) for s in channel_sums]
    std = [np.power((channel_squared_sums[s] / len(os.listdir(img_path)[:-1]) - mean[s] ** 2), [0.5])[0]
           for s in range(3)]

    return mean, std

def images_per_class(annotations_path):
    """
    Calculates the number of images per category that contains at least one of the category.
    :param      annotations_path: (string) Path to COCO annotations.
    :return     image_count: (list dict) The number of images that contains an object of a category, per category.
    """
    with open(annotations_path, 'r') as file:
        data = json.load(file)

    category_counts = []
    for img in data["images"]:
        annotations = []

        # For each image, extract the annotations corresponding to that image
        for ann in data["annotations"]:
            if ann["image_id"] == img["id"]:
                annotations.append(ann)

        # Keep track of one of each category that appear in the set of annotations
        category_counts.append(set([a["category_id"] for a in annotations]))

    # For each category in the dataset, count the number of images that have at least one object of that category
    image_counts = []
    for cat in data["categories"]:
        count = sum([1 for j in category_counts if cat["id"] in j])
        image_counts.append({"id": cat["id"], "image_count": count})

    return image_counts

def register_seeds(img_path, train_annotations=None, test_annotations=None, val_annotations=None):
    """
    Runs the register_coco_instances() function on any combination of training, testing, and validation datasets.
    :param      img_path: (string) Path to all images.
    :param      train_annotations: (string) Path to the training COCO annotations.
    :param      test_annotations: (string) Path to the testing COCO annotations.
    :param      val_annotations: (string) Path to the validation COCO annotations.
    """
    if train_annotations:
        register_coco_instances('orchid_train',
                                {
                                    "thing_classes": ['Viable', 'Non-Viable', 'Empty'],
                                    "thing_colors": [(255, 0, 0), (255, 255, 0), (0, 0, 0)],
                                    "class_image_count": images_per_class(train_annotations)
                                },
                                train_annotations, img_path)
    if test_annotations:
        register_coco_instances('orchid_test',
                                {
                                    "thing_classes": ['Viable', 'Non-Viable', 'Empty'],
                                    "thing_colors": [(255, 0, 0), (255, 255, 0), (0, 0, 0)],
                                    "class_image_count": images_per_class(test_annotations)
                                },
                                test_annotations, img_path)
    if val_annotations:
        register_coco_instances('orchid_val',
                                {
                                    "thing_classes": ['Viable', 'Non-Viable', 'Empty'],
                                    "thing_colors": [(255, 0, 0), (255, 255, 0), (0, 0, 0)],
                                    "class_image_count": images_per_class(val_annotations)
                                },
                                val_annotations, img_path)

def build_config(img_path, model_name="", **kwargs):
    # set up the logger
    setup_logger()
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

    cfg.OUTPUT_DIR = "model_weights"

    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.5
    cfg.DATALOADER.REPEAT_SQRT = True

    # set custom means and stds
    mean, std = mean_and_std(os.path.join(img_path))
    cfg.MODEL.PIXEL_MEAN = np.array(mean, dtype=float).tolist()
    cfg.MODEL.PIXEL_STD = np.array(std, dtype=float).tolist()

    cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]

    # # enable cascade mask r-cnn
    cfg.MODEL.ROI_HEADS.NAME = "CascadeROIHeads"
    # cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True

    # # standard parameters for machine learning
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 1
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0002
    cfg.SOLVER.MAX_ITER = 21000
    cfg.SOLVER.WARMUP_ITERS = int(cfg.SOLVER.MAX_ITER / 5)
    # cfg.SOLVER.STEPS = [2000, 3000]
    cfg.SOLVER.CHECKPOINT_PERIOD = 4600
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.WEIGHT_DECAY = 0.01

    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [
        [16, 32, 48, 64],
        [80, 96, 112, 128],
        [160, 192, 224, 256],
        [320, 384, 448, 512],
        [640, 768, 896, 1024]
    ]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]]
    cfg.MODEL.FPN.FUSE_TYPE = "avg"

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

    cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = True
    cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = True
    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER = 0.5
    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES = 3

    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.RPN.POSITIVE_FRACTION = 0.8

    # overlapping bounding box threshold for instances
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 3000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1500
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 3000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1500
    cfg.MODEL.RPN.NMS_THRESH = 0.5

    # ratio of foreground images
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    # number of foreground + background proposals
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.8

    # cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS = (
    #     (10.0, 10.0, 5.0, 5.0),
    #     (20.0, 20.0, 10.0, 10.0),
    #     (30.0, 30.0, 15.0, 15.0),
    #     (40.0, 40.0, 20.0, 20.0),
    # )
    # cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS = (0.5, 0.6, 0.7, 0.8)

    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
    # threshold for confidence
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # removing overlapping bounding boxes of the same class
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.INPUT.MIN_SIZE_TEST = 1600
    cfg.INPUT.MAX_SIZE_TEST = 1600

    # max number of instances per image
    cfg.TEST.DETECTIONS_PER_IMAGE = 800
    cfg.TEST.EVAL_PERIOD = 1000

    for param, value in kwargs.items():
        exec(param + " = " + str(value))

    if not model_name:
        date = datetime.datetime.now()
        model_name = "model_" + str(date.year) + str(date.month) + str(date.day)
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, model_name)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    return cfg