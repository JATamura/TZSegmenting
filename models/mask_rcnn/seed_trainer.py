import weakref
import os
import torch
import yaml
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, hooks
from detectron2.evaluation import DatasetEvaluators, COCOEvaluator, DatasetEvaluator
from detectron2.solver.build import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.engine import launch
from seed_evaluator import ClsAgnNMSEvaluator, ClassCountEvaluator, SeedSegmentationEvaluator
from models.mask_rcnn.configure_parameters import build_config

class SeedTrainer(DefaultTrainer):
    """
    Custom trainer used for the Cascade Mask R-CNN model that was trained.
    """
    @classmethod
    def build_train_loader(cls, cfg):

        # Applies flips, resizing, cropping, and randomisation of saturation, brightness, and contrast as data augmentation strategies.
        aug = [
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.ResizeShortestEdge(short_edge_length=[1400, 2000]),
            T.RandomCrop("absolute", (900, 900)),
            T.RandomSaturation(0.9, 1.1),
            T.RandomBrightness(0.9, 1.1),
            T.RandomContrast(0.9, 1.1),
        ]
        mapper = DatasetMapper(cfg, is_train=True, augmentations=aug)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        # Builds default COCO evaluator and custom evaluators for calculating performance along different axes
        coco_evaluator = COCOEvaluator(dataset_name, output_dir=output_folder, max_dets_per_image=800)
        nms_evaluator = ClsAgnNMSEvaluator(dataset_name, 0.5, max_dets_per_image=800, output_dir=output_folder)
        seed_seg_evaluator = SeedSegmentationEvaluator(dataset_name, 0.5, max_dets_per_image=800, output_dir=output_folder)
        mae_evaluator = ClassCountEvaluator(dataset_name, 0.5, max_dets_per_image=800, output_dir=output_folder)

        # Control which evaluators are needed by removing from this list
        evaluator_list = [
            coco_evaluator,
            nms_evaluator,
            seed_seg_evaluator,
            mae_evaluator
        ]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_optimizer(cls, cfg, model):

        # Builds learning rate optimisers that uses AdamW
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
    image_path = "../../datasets/dataset1/all_images"
    train = "../../datasets/dataset1/coco_format/post_quality_check/model_data/train.json"
    test = "../../datasets/dataset1/coco_format/post_quality_check/model_data/test.json"
    val = "../../datasets/dataset1/coco_format/post_quality_check/model_data/val.json"

    param_dict = {}

    cfg = build_config(image_path, train, test, val, param_dict)
    cfg.MODEL.WEIGHTS = "models/mask_rcnn/model_weights/output_06_25_cascade_16000/model_final.pth"
    model_config = yaml.safe_load(cfg.dump())
    print(model_config)
    with open('config.yaml', 'w') as file:
        yaml.dump(model_config, file)

    # Build config and trainer
    cfg = build_config(image_path, train, test, val, param_dict)
    trainer = SeedTrainer(cfg)
    trainer.resume_or_load(resume=False)
    checkpointer = DetectionCheckpointer(
        trainer.build_model(cfg),
        cfg.OUTPUT_DIR,
        trainer=weakref.proxy(trainer),
    )

    # Add a hook to save the best performing set of weights
    trainer.register_hooks(
        [hooks.BestCheckpointer(eval_period=cfg.TEST.EVAL_PERIOD, checkpointer=checkpointer, val_metric="bbox/AP50")]
    )

    return trainer.train()

if __name__ == "__main__":
    # main()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    launch(main, torch.cuda.device_count())