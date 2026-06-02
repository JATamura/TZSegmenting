import json
import os

import torch

from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import inference_on_dataset, DatasetEvaluators
from model_training_and_evaluation_methods.mask_rcnn.configure_parameters import register_seeds
from model_training_and_evaluation_methods.mask_rcnn.seed_evaluator import ClassCountEvaluator
from utils import REPO_PATH


def main():
    images_in_analysis_and_test_set = ['086.jpg', '126.jpg', '141.jpg', '156.jpg', '196.jpg', '221.jpg', '246.jpg', '256.jpg', '281.jpg', '296.jpg',
                                       '321.jpg', '356.jpg', '371.jpg', '376.jpg', '391.jpg', '466.jpg', '491.jpg', '496.jpg']

    image_path = os.path.join(REPO_PATH, "datasets/dataset1/all_images")
    test = os.path.join(REPO_PATH, "datasets/dataset1/coco_format/post_quality_check/model_data/test.json")
    with open(test, 'r') as file:
        data = json.load(file)
        for i in data['images'][:]:
            if i['file_name'] not in images_in_analysis_and_test_set:
                data['images'].remove(i)

        image_ids = [i['id'] for i in data['images']]
        for a in data['annotations'][:]:
            if a['image_id'] not in image_ids:
                data['annotations'].remove(a)

    with open("post_quality_check/annotations_images_in_analysis_and_test_set.json", "w") as f:
        json.dump(data, f, indent=4)

    os.chdir(REPO_PATH)
    train = os.path.join(REPO_PATH, "datasets/dataset1/coco_format/post_quality_check/model_data/train.json")
    register_seeds(image_path, train_annotations=train, test_annotations=os.path.join(REPO_PATH, 'agreement_analysis', 'post_quality_check',
                                                                                      "annotations_images_in_analysis_and_test_set.json"))

    config_path = "model_results_and_final_weights/final_tz_segmentor/config.yaml"

    print("Getting model weights")

    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
        print("No GPU available, using CPU")
    else:
        cfg.MODEL.DEVICE = "cuda"
        print("Using GPU")

    predictor = DefaultPredictor(cfg)
    model = predictor.model
    dataset_name = "orchid_test"

    data_loader = build_detection_test_loader(cfg, dataset_name)
    mae_evaluator = ClassCountEvaluator(dataset_name, 0.7, 800)

    print("Running evaluation")
    results = inference_on_dataset(model, data_loader,
                                   DatasetEvaluators([
                                       mae_evaluator,
                                   ]))
    with open("agreement_analysis/post_quality_check/evaluation_metrics_on_images_in_analysis_and_test_set.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()
