# seed_evaluator has methods for adding more evaluation metrics
import json
import os
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torchvision.ops import nms
from sklearn.metrics import mean_absolute_error
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import DatasetEvaluators, COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.structures import Instances
from utils import REPO_PATH


class ClsAgnNMSEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cls_agnostic_nms, max_dets_per_image, output_dir=None):
        super().__init__(dataset_name, output_dir=output_dir, max_dets_per_image=max_dets_per_image)
        self.cls_agnostic_nms = cls_agnostic_nms

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            instances = output["instances"].to(self._cpu_device)
            nms_indices = nms(instances._fields["pred_boxes"].tensor,
                              instances._fields["scores"], self.cls_agnostic_nms)
            nms_prediction = {'instances': Instances(image_size=instances.image_size,
                                                     pred_boxes=instances.pred_boxes[nms_indices],
                                                     scores=instances.scores[nms_indices],
                                                     pred_classes=instances.pred_classes[nms_indices],
                                                     pred_masks=instances.pred_masks[nms_indices])}

            prediction["instances"] = instances_to_coco_json(nms_prediction["instances"], input["image_id"])
            self._predictions.append(prediction)

    def evaluate(self, img_ids=None):
        results = {}
        for k, v in super().evaluate().items():
            results[k + "_cls_agn_nms"] = v
        return results


class SeedSegmentationEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cls_agnostic_nms, max_dets_per_image, output_dir=None):
        super().__init__(dataset_name, output_dir=output_dir, max_dets_per_image=max_dets_per_image)
        self.cls_agnostic_nms = cls_agnostic_nms

        print(self._metadata)
        self._coco_api.dataset["categories"] = [{'id': 1, 'name': 'Seed', 'supercategory': ''}]
        for ann in self._coco_api.dataset["annotations"]:
            ann["category_id"] = 1
        self._coco_api.createIndex()

        self._metadata = {"thing_classes": ["Seed"]}

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            instances = output["instances"].to(self._cpu_device)

            nms_indices = nms(instances._fields["pred_boxes"].tensor,
                              instances._fields["scores"], self.cls_agnostic_nms)
            nms_prediction = {'instances': Instances(image_size=instances.image_size,
                                                     pred_boxes=instances.pred_boxes[nms_indices],
                                                     scores=instances.scores[nms_indices],
                                                     pred_classes=Tensor([1 for i in instances.pred_classes[nms_indices]]),
                                                     pred_masks=instances.pred_masks[nms_indices])}

            prediction["instances"] = instances_to_coco_json(nms_prediction["instances"], input["image_id"])
            self._predictions.append(prediction)

    def evaluate(self, img_ids=None):
        results = {}
        for k, v in super().evaluate().items():
            results[k + "_seed_class"] = v
        return results


class ClassCountEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cls_agnostic_nms, max_dets_per_image, output_dir=None):
        super().__init__(dataset_name, output_dir=output_dir, max_dets_per_image=max_dets_per_image)
        self.cls_agnostic_nms = cls_agnostic_nms
        self.actual = []
        self.pred = []
        self.nms_pred = []

    def reset(self):
        super().reset()
        self.actual = []
        self.pred = []
        self.nms_pred = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            annotations = self._coco_api.loadAnns(self._coco_api.getAnnIds(imgIds=input["image_id"]))
            actual_classes = [annotation["category_id"] for annotation in annotations]
            self.actual.append([
                actual_classes.count(1),
                actual_classes.count(2),
                actual_classes.count(3),
                len(actual_classes)
            ])
            if self.actual[-1][1]:
                self.actual[-1].append(self.actual[-1][0] / self.actual[-1][1])
            elif self.actual[-1][0]:
                self.actual[-1].append(1)
            else:
                self.actual[:1].append(0)

            predicted_classes = output["instances"].pred_classes.tolist()
            self.pred.append([
                predicted_classes.count(0),
                predicted_classes.count(1),
                predicted_classes.count(2),
                len(predicted_classes)
            ])
            if self.pred[-1][1]:
                self.pred[-1].append(self.pred[-1][0] / self.pred[-1][1])
            elif self.pred[-1][0]:
                self.pred[-1].append(1)
            else:
                self.pred[-1].append(0)

            instances = output["instances"].to(self._cpu_device)
            nms_indices = nms(instances._fields["pred_boxes"].tensor,
                              instances._fields["scores"], self.cls_agnostic_nms)
            nms_prediction = {'instances': Instances(image_size=instances.image_size,
                                                     pred_boxes=instances.pred_boxes[nms_indices],
                                                     scores=instances.scores[nms_indices],
                                                     pred_classes=instances.pred_classes[nms_indices],
                                                     pred_masks=instances.pred_masks[nms_indices])}
            nms_classes = nms_prediction["instances"].pred_classes.tolist()
            self.nms_pred.append([
                nms_classes.count(0),
                nms_classes.count(1),
                nms_classes.count(2),
                len(nms_classes)
            ])
            if self.nms_pred[-1][1]:
                self.nms_pred[-1].append(self.nms_pred[-1][0] / self.nms_pred[-1][1])
            elif self.nms_pred[-1][0]:
                self.nms_pred[-1].append(1)
            else:
                self.nms_pred[-1].append(0)

    def evaluate(self, img_ids=None):
        predicted = np.sum(self.pred, axis=0) / len(self.pred)
        nms_predicted = np.sum(self.nms_pred, axis=0) / len(self.nms_pred)

        maes = np.array(mean_absolute_error(self.actual, self.pred, multioutput="raw_values"))
        nms_maes = np.array(mean_absolute_error(self.actual, self.nms_pred, multioutput="raw_values"))
        results = {
            "base_raw_values": {
                "viable": predicted[0],
                "non_viable": predicted[1],
                "empty": predicted[2],
                "total": predicted[3],
                "viability_ratio": predicted[4],
            },
            "nms_raw_values": {
                "viable": nms_predicted[0],
                "non_viable": nms_predicted[1],
                "empty": nms_predicted[2],
                "total": nms_predicted[3],
                "viability_ratio": nms_predicted[4],
            },
            "base_maes": {
                "viable": maes[0],
                "non_viable": maes[1],
                "empty": maes[2],
                "total": maes[3],
                "viability_ratio": maes[4],
            },
            "nms_maes": {
                "viable": nms_maes[0],
                "non_viable": nms_maes[1],
                "empty": nms_maes[2],
                "total": nms_maes[3],
                "viability_ratio": nms_maes[4]
            }
        }
        return results


def run_on_individual_test_data():
    from configure_parameters import register_seeds
    image_path = os.path.join(REPO_PATH, "datasets/dataset1/all_images")
    train = os.path.join(REPO_PATH, "datasets/dataset1/coco_format/post_quality_check/model_data/train.json")
    test = os.path.join(REPO_PATH, "datasets/dataset1/coco_format/post_quality_check/model_data/test.json")
    val = os.path.join(REPO_PATH, "datasets/dataset1/coco_format/post_quality_check/model_data/val.json")

    # First make test jsons to register instances for individual images
    test_json = json.load(open(test, 'r'))
    for i in test_json['images']:
        new_test_json = test_json.copy()
        new_test_json['images'] = [i]
        id = i['id']
        new_annotations = []
        for a in test_json['annotations']:
            if a['image_id'] == id:
                new_annotations.append(a)
        new_test_json['annotations'] = new_annotations
        json.dump(new_test_json,
                  open(os.path.join(REPO_PATH, f"datasets/dataset1/coco_format/post_quality_check/model_data/single_images/{id}_test.json"), 'w'),
                  indent=4)

    per_image_results = []
    for i in test_json['images']:
        id = i['id']
        test_path = os.path.join(os.path.join(REPO_PATH, f"datasets/dataset1/coco_format/post_quality_check/model_data/single_images/{id}_test.json"))
        # Register dataset for individual image. needs new name setting too.
        dataset_name = f'orchid_test_{id}'
        register_seeds(image_path, train, test_path, val, dataset_name)

        os.chdir(REPO_PATH)
        config_path = "model_results_and_final_weights/final_tz_segmentor/config.yaml"

        print("Getting model weights")

        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        if not torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cpu"
        else:
            cfg.MODEL.DEVICE = "cuda"

        predictor = DefaultPredictor(cfg)
        model = predictor.model
        # Make sure model is in eval mode
        model.eval()

        data_loader = build_detection_test_loader(cfg, dataset_name)
        coco_evaluator = COCOEvaluator(dataset_name, max_dets_per_image=800)
        nms_evaluator = ClsAgnNMSEvaluator(dataset_name, 0.7, 800)
        mae_evaluator = ClassCountEvaluator(dataset_name, 0.7, 800)
        seed_seg_evaluator = SeedSegmentationEvaluator(dataset_name, 0.7, 800)
        data_evaluators = DatasetEvaluators([
            coco_evaluator,
            nms_evaluator,
            seed_seg_evaluator,
            mae_evaluator,
        ])

        with torch.no_grad():
            print("Running evaluation")
            results = inference_on_dataset(model, data_loader, data_evaluators)
            # print(results)

            image_name = os.path.basename(i['file_name'])

            ap_results = [results['segm_cls_agn_nms']['AP']] + [results['segm_cls_agn_nms']['AP-' + c] for c in ['Viable', 'Non-Viable', 'Empty']] + [
                results['segm_seed_class']['AP']]
            mae_scores = [results['nms_maes'][c] for c in ['viable', 'non_viable', 'empty', 'total']]

            raw_counts = [results['nms_raw_values'][c] for c in ['viable', 'non_viable', 'empty', 'total']]

            per_image_results.append([image_name] + ap_results + mae_scores + raw_counts)
            out_df = pd.DataFrame(per_image_results,
                                  columns=['image_name'] + ['AP', 'AP-Viable', 'AP-Non-Viable', 'AP-Empty', 'AP-Seed'] + ['MAE-Viable',
                                                                                                                          'MAE-Non-Viable',
                                                                                                                          'MAE-Empty',
                                                                                                                          'MAE-Total'] + [
                                              'Predicted Viable',
                                              'Predicted Non-Viable',
                                              'Predicted Empty', 'Predicted Total'])
            out_df.to_csv("model_results_and_final_weights/final_tz_segmentor/test_results_on_individual_images.csv", index=False)

    out_df.to_csv("model_results_and_final_weights/final_tz_segmentor/test_results_on_individual_images.csv")


def run_on_test_data():
    from configure_parameters import register_seeds
    image_path = os.path.join(REPO_PATH, "datasets/dataset1/all_images")
    train = os.path.join(REPO_PATH, "datasets/dataset1/coco_format/post_quality_check/model_data/train.json")
    test = os.path.join(REPO_PATH, "datasets/dataset1/coco_format/post_quality_check/model_data/test.json")
    val = os.path.join(REPO_PATH, "datasets/dataset1/coco_format/post_quality_check/model_data/val.json")
    register_seeds(image_path, train, test, val)

    os.chdir(REPO_PATH)
    config_path = "model_results_and_final_weights/final_tz_segmentor/config.yaml"

    print("Getting model weights")

    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    else:
        cfg.MODEL.DEVICE = "cuda"

    predictor = DefaultPredictor(cfg)
    model = predictor.model
    # Make sure model is in eval mode
    model.eval()

    dataset_name = "orchid_test"

    data_loader = build_detection_test_loader(cfg, dataset_name)
    coco_evaluator = COCOEvaluator(dataset_name, max_dets_per_image=800)
    nms_evaluator = ClsAgnNMSEvaluator(dataset_name, 0.7, 800)
    mae_evaluator = ClassCountEvaluator(dataset_name, 0.7, 800)
    seed_seg_evaluator = SeedSegmentationEvaluator(dataset_name, 0.7, 800)
    data_evaluators = DatasetEvaluators([
        coco_evaluator,
        nms_evaluator,
        seed_seg_evaluator,
        mae_evaluator,
    ])
    with torch.no_grad():
        print("Running evaluation")
        results = inference_on_dataset(model, data_loader, data_evaluators)
    print(results)

    with open("model_results_and_final_weights/final_tz_segmentor/final_evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=4)


def main():
    # run_on_test_data()
    run_on_individual_test_data()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No GPU available, using CPU")

    main()
