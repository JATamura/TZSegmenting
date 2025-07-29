import json
import os
import random
import shutil
import copy
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
import pandas as pd
from sklearn.model_selection import train_test_split
from ultralytics.data.converter import convert_coco
import yaml
import logging
import cv2
from pycocotools import mask as cocomask

# Source: https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#create-custom-coco-dataset
def rle_to_coco(annotation: dict) -> list[dict]:
    """Transform the rle coco annotation (a single one) into coco style.
    In this case, one mask can contain several polygons, later leading to several `Annotation` objects.
    In case of not having a valid polygon (the mask is a single pixel) it will be an empty list.
    Parameters
    ----------
    annotation : dict
        rle coco style annotation
    Returns
    -------
    list[dict]
        list of coco style annotations (in dict format)
    """

    annotation["segmentation"] = cocomask.frPyObjects(
        annotation["segmentation"],
        annotation["segmentation"]["size"][0],
        annotation["segmentation"]["size"][1],
    )

    masked_arr = cocomask.decode(annotation["segmentation"])
    contours, _ = cv2.findContours(masked_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []

    for contour in contours:
        if contour.size >= 6:
            segmentation.append(contour)

    if len(segmentation) == 0:
        logging.debug(
            f"Annotation with id {annotation['id']} is not valid, it has no segmentations."
        )
        annotations = []

    else:
        annotations = list()
        for i, seg in enumerate(segmentation):

            single_annotation = copy.deepcopy(annotation)
            single_annotation["segmentation_coords"] = (
                seg.astype(float).flatten().tolist()
            )
            single_annotation["bbox"] = list(cv2.boundingRect(seg))
            single_annotation["area"] = cv2.contourArea(seg)
            single_annotation["instance_id"] = annotation["id"]
            single_annotation["annotation_id"] = f"{annotation['id']}_{i}"

            annotations.append(single_annotation)

    return annotations

def combine_datasets(path_to_annotations, output_path):
    """
    Used to combine the five part dataset into one.
    :param      path_to_annotations (array string): File paths to COCO annotations
    :param      output_path (string): Output path.
    :return:
    """
    all_annotations = []
    for a in path_to_annotations:
        with open(a, 'r') as file:
            all_annotations.append(json.load(file))

    combined = {"licenses": [{"name": "", "id": 0, "url": ""}],
                "info": {"contributor": "", "date_created": "",
                         "description": "", "url": "", "version": "", "year": ""},
                "categories": [{"id": 1, "name": "Viable", "supercategory": ""},
                               {"id": 2, "name": "Non-Viable", "supercategory": ""},
                               {"id": 3, "name": "Empty", "supercategory": ""}],
                "images": [],
                "annotations": []
                }

    image_id = 0
    annotation_id = 0
    for annotations in all_annotations:
        for image in annotations["images"]:
            combined["images"].append(copy.deepcopy(image))
            new_image_id = image_id + image["id"] - 1
            combined["images"][new_image_id]["id"] = image_id + image["id"]
        for annotation in annotations["annotations"]:
            combined["annotations"].append(copy.deepcopy(annotation))
            new_annotation_id = annotation_id + annotation["id"] - 1
            combined["annotations"][new_annotation_id]["id"] = annotation_id + annotation["id"]
            combined["annotations"][new_annotation_id]["image_id"] = image_id + annotation["image_id"]
            if type(combined["annotations"][new_annotation_id]["segmentation"]) != list:
                new_seg = rle_to_coco(combined["annotations"][new_annotation_id])[0]
                combined["annotations"][new_annotation_id]["segmentation"] = [new_seg["segmentation_coords"]]

        image_id += len(annotations["images"])
        annotation_id += len(annotations["annotations"])

    with open(output_path, "w") as outfile:
        json.dump(combined, outfile)

def extract_annotations(all_annotations, image_id):
    """
    Returns all annotations from an image using the image ID in the COCO dataset.
    :param      all_annotations (dict): COCO dataset.
    :param      image_id (int): Image ID.
    :return     image_annotations (array dict): All annotations extracted from the specified image.
    """
    image_annotations = []
    for annotation in all_annotations:
        if annotation["image_id"] == image_id:
            image_annotations.append(annotation)
    return image_annotations

def avg_seg_size(annotations):
    """
    Calculates the average segmention/mask size. Used to stratify the dataset.
    :param     annotations (dict array): Array of annotations in COCO format (usually from a single image).
    :return    avg_area (float): Average segmentation/mask size of annotations.
    """
    areas = 0
    for a in annotations:
        a_seg = np.array(a["segmentation"][0])
        polygon = Polygon(a_seg.reshape(int(len(a_seg) / 2), 2))
        areas += polygon.area
    avg_seg = areas / len(annotations)
    return avg_seg

def avg_bbox_size(annotations):
    """
    Calculates the average bbox size. Primarily used to gauge Mask R-CNN anchor sizes.
    :param     annotations (dict array): Array of annotations in COCO format (usually from a single image).
    :return    avg_bbox (float): Average bbox size of annotations.
    """
    areas = 0
    for a in annotations:
        areas += a["bbox"][2] * a["bbox"][3]
    avg_bbox = areas / len(annotations)
    return avg_bbox

def seed_stats(dataset):
    """
    Calculate image-by-image statistics given a COCO dataset.
    :param      dataset (dict): COCO dataset.
    :return     stats (pandas.DataFrame): Image-by-image statistics.
    """

    file_names = []
    viable = []
    nonviable = []
    empty = []
    total_counts = []
    viable_ratios = []
    avg_sizes = []
    avg_bboxes = []
    for image in dataset["images"]:
        annotations = extract_annotations(dataset["annotations"], image["id"])
        if len(annotations) > 0:
            file_names.append(image["file_name"])
            viable.append(len([a for a in annotations if a["category_id"] == 1]))
            nonviable.append(len([a for a in annotations if a["category_id"] == 2]))
            empty.append(len([a for a in annotations if a["category_id"] == 3]))
            total_counts.append(len(annotations))
            viable_ratios.append(len([a for a in annotations if a["category_id"] == 1]) / len(annotations))
            avg_sizes.append(avg_seg_size(annotations))
            avg_bboxes.append(avg_bbox_size(annotations))

    total_bin = pd.qcut(total_counts, q=4, labels=False, duplicates='drop')
    viable_bin = pd.qcut(viable, q=3, labels=False, duplicates='drop')
    size_bin = pd.qcut(avg_sizes, q=4, labels=False, duplicates='drop')

    stats = pd.DataFrame(
        {"file_names": file_names,
         "viable": viable,
         "nonviable": nonviable,
         "empty": empty,
         "total": total_counts,
         "viable_ratio": viable_ratios,
         "avg_sizes": np.sqrt(avg_sizes),
         "avg_bboxes": np.sqrt(avg_bboxes),
         "total_bin": total_bin,
         "viable_bin": viable_bin,
         "size_bin": size_bin}
    )

    # # Creates graph to visualise seed count histogram.
    # plt.hist(stats["total"], bins=20, edgecolor='black')
    # plt.xlabel('Total seed count')
    # plt.ylabel('Number of images')
    # plt.show()

    return stats


def stratify(stats, test_ratio=0.2, val_ratio=0.2):
    """
    Stratify the dataset using specific numerical values. Currently, uses the number of viable seeds and the average seed size in each image.
    :param      stats (pandas.DataFrame): Image-by-image statistics of the dataset.
    :param      test_ratio (float): Ratio of the dataset to split into the test set used for calculating final model performance.
    :param      val_ratio (float): Ratio of the dataset to split into the validation set used for evaluating performance during training.
    :return     train (pandas.DataFrame): Statistics of the images stratified into the training dataset.
    :return     test (pandas.DataFrame): Statistics of the images stratified into the testing dataset.
    :return     val (pandas.DataFrame): Statistics of the images stratified into the validation dataset.
    """
    stats["stratify"] = (
        # stats["total_bin"].astype(str) + "_" +
        stats["viable_bin"].astype(str) + "_" +
        stats["size_bin"].astype(str)
    )

    train, test_and_val = train_test_split(stats, test_size=test_ratio+val_ratio,
                                           random_state=0, stratify=stats["stratify"])

    test, val = train_test_split(test_and_val, test_size=val_ratio/(test_ratio+val_ratio),
                                 random_state=0, stratify=test_and_val["stratify"])

    return train, test, val

# Splitting the dataset
def train_test_split_coco(path_to_annotations, output_path, val_ratio=0.2, test_ratio=0.2,
                          val_imgs=[], test_imgs=[], random_seed=None):

    if test_imgs is None:
        test_imgs = []
    if val_imgs is None:
        val_imgs = []
    if random_seed != None:
        random.seed(random_seed)

    # load annotations
    with open(path_to_annotations, 'r') as file:
        all = json.load(file)

    # list all file names
    all_imgs = []
    for img in all["images"]:
        all_imgs.append(img["file_name"])
    train_imgs = copy.deepcopy(all_imgs)

    # randomly select test and validation images if no lists were given
    if len(test_imgs) == 0:
        for i in range(int(len(all_imgs) * test_ratio)):
            img = random.choice(train_imgs)
            while img in test_imgs:
                img = random.choice(train_imgs)
            test_imgs.append(img)
    if len(val_imgs) == 0:
        for i in range(int(len(all_imgs) * val_ratio)):
            img = random.choice(train_imgs)
            while img in val_imgs:
                img = random.choice(train_imgs)
            val_imgs.append(img)

    for img in test_imgs + val_imgs:
        train_imgs.remove(img)

    train = copy.deepcopy(all)
    test = copy.deepcopy(all)
    val = copy.deepcopy(all)

    for img in all["images"]:
        if not img["file_name"] in train_imgs:
            train["images"].remove(img)
        if not img["file_name"] in test_imgs:
            test["images"].remove(img)
        if not img["file_name"] in val_imgs:
            val["images"].remove(img)

    # remove the appropriate annotations from the json file
    train_ids = [img["id"] for img in train["images"]]
    test_ids = [img["id"] for img in test["images"]]
    val_ids = [img["id"] for img in val["images"]]
    for a in all["annotations"]:
        if not a["image_id"] in train_ids:
            train["annotations"].remove(a)
        if not a["image_id"] in test_ids:
            test["annotations"].remove(a)
        if not a["image_id"] in val_ids:
            val["annotations"].remove(a)

    return train, test, val

def convert_iscrowd(dataset_path):
    """
    Converts all iscrowd values to 0. Specific to the seeds dataset as seeds were never annotated in groups.
    :param      dataset_path (string): Directory of the COCO dataset.
    :return:
    """
    with open(dataset_path, 'r') as file:
        data = json.load(file)
    for ann in data["annotations"]:
        ann["iscrowd"] = 0
    with open(dataset_path, 'w') as file:
        json.dump(data, file)

if __name__ == "__main__":

    # Merge five part datasets into one

    print("Combining datasets")

    datasets_to_merge = ["../datasets/dataset1/coco/pre_quality_check/part1_preqc.json",
                         "../datasets/dataset1/coco/pre_quality_check/part2_preqc.json",
                         "../datasets/dataset1/coco/pre_quality_check/part3_preqc.json",
                         "../datasets/dataset1/coco/pre_quality_check/part4_preqc.json",
                         "../datasets/dataset1/coco/pre_quality_check/part5_preqc.json"]
    output_path = "../datasets/dataset1/coco/pre_quality_check/all_preqc.json"
    combine_datasets(datasets_to_merge, output_path)
    convert_iscrowd(output_path)

    datasets_to_merge = ["../datasets/dataset1/coco/post_quality_check/part1_postqc.json",
                         "../datasets/dataset1/coco/post_quality_check/part2_postqc.json",
                         "../datasets/dataset1/coco/post_quality_check/part3_postqc.json",
                         "../datasets/dataset1/coco/post_quality_check/part4_postqc.json",
                         "../datasets/dataset1/coco/post_quality_check/part5_postqc.json"]
    output_path = "../datasets/dataset1/coco/post_quality_check/all_postqc.json"
    combine_datasets(datasets_to_merge, output_path)
    convert_iscrowd(output_path)

    # ...and corresponding checks (duplicate datasets used for agreement analysis)
    datasets_to_merge = ["../datasets/dataset1/coco/check_1/part1_check1.json",
                         "../datasets/dataset1/coco/check_1/part2_check1.json",
                         "../datasets/dataset1/coco/check_1/part3_check1.json",
                         "../datasets/dataset1/coco/check_1/part4_check1.json",
                         "../datasets/dataset1/coco/check_1/part5_check1.json"]
    output_path = "../datasets/dataset1/coco/check_1/all_check1.json"
    combine_datasets(datasets_to_merge, output_path)
    convert_iscrowd(output_path)

    datasets_to_merge = ["../datasets/dataset1/coco/check_2/part1_check2.json",
                         "../datasets/dataset1/coco/check_2/part2_check2.json",
                         "../datasets/dataset1/coco/check_2/part3_check2.json",
                         "../datasets/dataset1/coco/check_2/part4_check2.json",
                         "../datasets/dataset1/coco/check_2/part5_check2.json"]
    output_path = "../datasets/dataset1/coco/check_2/all_check2.json"
    combine_datasets(datasets_to_merge, output_path)
    convert_iscrowd(output_path)

    # ----------------------------------------------------------------------------

    # Calculate seed stats, stratify accordingly

    dataset_path = [
        "../datasets/dataset1/coco/pre_quality_check/all_preqc.json",
        "../datasets/dataset1/coco/post_quality_check/all_postqc.json"
    ]

    print("Loading full dataset")
    with open(dataset_path[1], 'r') as file:
        post_qc_dataset = json.load(file)

    # Remove appropriate images from the json file
    imgs_with_issues = []
    for img in post_qc_dataset["images"]:
        # Remove images with excess number of seeds
        annotations = extract_annotations(post_qc_dataset["annotations"], img["id"])
        if len(annotations) > 800:
            imgs_with_issues.append((img["file_name"], len(annotations), img))
        # Remove 2 images with dark seed coats manually
        if img["file_name"] == "425.jpg":
            imgs_with_issues.append((img["file_name"], len(annotations), img))
        if img["file_name"] == "481.jpg":
            imgs_with_issues.append((img["file_name"], len(annotations), img))

    for file_name, object_count, img in imgs_with_issues:
        print(file_name + " was removed. It had " + str(object_count) + " objects.")
        post_qc_dataset["images"].remove(img)

    # Stratify according to the post quality checked dataset
    print("Stratifying data")
    stats = seed_stats(post_qc_dataset)
    train, test, val = stratify(stats)
    print("train size: " + str(len(train)))
    print("test size: " + str(len(test)))
    print("val size: " + str(len(val)))

    print("Finished stratification")

    # -----------------------------------------------------------------------------

    # Split the pre and qpost quality checked datasets using the stratified output, create train, test, and val datasets for COCO and YOLO

    coco_dir = [
        "../datasets/dataset1/coco/preqc_model_data",
        "../datasets/dataset1/coco/postqc_model_data"
    ]
    yolo_dir = [
        "../datasets/dataset1/yolo/pre_qc",
        "../datasets/dataset1/yolo/post_qc"
    ]

    for i in range(2):
        if not os.path.exists(coco_dir[i]):
            os.makedirs(coco_dir[i], exist_ok=True)
        print("Splitting dataset: " + coco_dir[i])
        # Splitting the data
        train_set, val_set, test_set = train_test_split_coco(
            dataset_path[i], coco_dir[i],
            test_imgs=list(test["file_names"].to_dict().values()),
            val_imgs=list(val["file_names"].to_dict().values()),
            random_seed=42
        )

        # Save the COCO datasets as json files
        with open(os.path.join(coco_dir[i], "train.json"), "w") as outfile:
            json.dump(train_set, outfile)
        with open(os.path.join(coco_dir[i], "test.json"), "w") as outfile:
            json.dump(test_set, outfile)
        with open(os.path.join(coco_dir[i], "val.json"), "w") as outfile:
            json.dump(val_set, outfile)

        # Convert the COCO datasets to YOLO
        print("Converting dataset " + str(i) + " to YOLO")
        convert_coco(coco_dir[i], yolo_dir[i], use_segments=True, cls91to80=False)

        # Create directories for YOLO datasets
        for dir in os.listdir(os.path.join(yolo_dir[i], "labels")):
            if not os.path.exists(os.path.join(yolo_dir[i], "images", dir)):
                os.makedirs(os.path.join(yolo_dir[i], "images", dir), exist_ok=True)
            for txt in os.listdir(os.path.join(yolo_dir[i], "labels", dir)):
                img = txt.split(".txt")[0] + ".jpg"
                shutil.copy(os.path.join("../datasets/dataset1/all_images", img),
                            os.path.join(yolo_dir[i], "images", dir, img))

        # Create yaml files and save to respective directories
        yolo_yaml = {
            "names": {
                0: "Viable",
                1: "Non-Viable",
                2: "Empty"
            },
            "path": yolo_dir[i],
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
        }
        with open(os.path.join(yolo_dir[i], 'data.yaml'), 'w') as file:
            yaml.dump(yolo_yaml, file)

    print("Complete")

