import json
import os
import random
import shutil
import copy
import numpy as np
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
    """Transform the rle coco_format annotation (a single one) into coco_format style.
    In this case, one mask can contain several polygons, later leading to several `Annotation` objects.
    In case of not having a valid polygon (the mask is a single pixel) it will be an empty list.
    Parameters
    ----------
    annotation : dict
        rle coco_format style annotation
    Returns
    -------
    list[dict]
        list of coco_format style annotations (in dict format)
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

def combine_preqc_datasets(path_to_annotations, output_path):
    """
    Used to combine the five part pre-quality checked dataset into one.
    :param      path_to_annotations: (list string) File paths to the 5 part pre-quality checked COCO datasets.
    :param      output_path: (string) Output path.
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

def split_postqc_dataset(annotations_path, output_path):
    """
    Used to split the post-quality checked dataset with the duplicate annotations for agreement analysis into their respective files.
    :param      annotations_path: (string) Path to the post-quality checked COCO dataset.
    :param      output_path: (string) Path to where the split up dataset will be stored as json files.
    """

    with open(annotations_path, 'r') as file:
        data = json.load(file)

    base_imgs = []
    dupe_1_imgs = []
    dupe_2_imgs = []

    for img in data["images"]:
        if "_1" in img["file_name"]:
            img["file_name"] = img["file_name"].replace("_1", "")
            dupe_1_imgs.append(img)
        elif "_2" in img["file_name"]:
            img["file_name"] = img["file_name"].replace("_2", "")
            dupe_2_imgs.append(img)
        else:
            base_imgs.append(img)

    base_anns = []
    dupe_1_anns = []
    dupe_2_anns = []
    for ann in data["annotations"]:
        if type(ann["segmentation"]) != list:
            ann["segmentation"] = [rle_to_coco(ann)[0]["segmentation_coords"]]
        if ann["image_id"] in [image["id"] for image in dupe_1_imgs]:
            dupe_1_anns.append(ann)
        elif ann["image_id"] in [image["id"] for image in dupe_2_imgs]:
            dupe_2_anns.append(ann)
        else:
            base_anns.append(ann)

    default = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {"contributor": "", "date_created": "",
                 "description": "", "url": "", "version": "", "year": ""},
        "categories": [{"id": 1, "name": "Viable", "supercategory": ""},
                       {"id": 2, "name": "Non-Viable", "supercategory": ""},
                       {"id": 3, "name": "Empty", "supercategory": ""}],
        "images": [],
        "annotations": []
    }

    base_dataset = copy.deepcopy(default)
    base_dataset["images"] = base_imgs
    base_dataset["annotations"] = base_anns
    for i, image in enumerate(base_dataset["images"]):
        for annotation in base_dataset["annotations"]:
            if annotation["image_id"] == image["id"]:
                annotation["image_id"] = i+1
        image["id"] = i+1
    for i, annotation in enumerate(base_dataset["annotations"]):
        annotation["id"] = i+1
    with open(os.path.join(output_path, "base_dataset.json"), 'w') as file:
        json.dump(base_dataset, file)

    dupe_1_dataset = copy.deepcopy(default)
    dupe_1_dataset["images"] = dupe_1_imgs
    dupe_1_dataset["annotations"] = dupe_1_anns
    for i, image in enumerate(dupe_1_dataset["images"]):
        for annotation in dupe_1_dataset["annotations"]:
            if annotation["image_id"] == image["id"]:
                annotation["image_id"] = i+1
        image["id"] = i+1
    for i, annotation in enumerate(dupe_1_dataset["annotations"]):
        annotation["id"] = i+1
    with open(os.path.join(output_path, "analysis_duplicate_1.json"), 'w') as file:
        json.dump(dupe_1_dataset, file)

    dupe_2_dataset = copy.deepcopy(default)
    dupe_2_dataset["images"] = dupe_2_imgs
    dupe_2_dataset["annotations"] = dupe_2_anns
    for i, image in enumerate(dupe_2_dataset["images"]):
        for annotation in dupe_2_dataset["annotations"]:
            if annotation["image_id"] == image["id"]:
                annotation["image_id"] = i+1
        image["id"] = i+1
    for i, annotation in enumerate(dupe_2_dataset["annotations"]):
        annotation["id"] = i+1
    with open(os.path.join(output_path, "analysis_duplicate_2.json"), 'w') as file:
        json.dump(dupe_2_dataset, file)

def extract_annotations(all_annotations, image_id):
    """
    Returns all annotations from an image using the image ID in the COCO dataset.
    :param      all_annotations: (dict) COCO dataset.
    :param      image_id: (int) Image ID.
    :return     image_annotations: (list dict) All annotations extracted from the specified image.
    """

    image_annotations = []
    for annotation in all_annotations:
        if annotation["image_id"] == image_id:
            image_annotations.append(annotation)
    return image_annotations

def avg_bbox_size(annotations):
    """
    Calculates the average bbox size. Primarily used to gauge Mask R-CNN anchor sizes.
    :param     annotations: (dict list) List of annotations in COCO format (usually from a single image).
    :return    avg_bbox: (float) Average bbox size of annotations.
    """

    if len(annotations) > 0:
        areas = 0
        for a in annotations:
            areas += a["bbox"][2] * a["bbox"][3]
        avg_bbox = areas / len(annotations)
    else:
        avg_bbox = 0
    return avg_bbox

def avg_seg_size(annotations):
    """
    Calculates the average segmention/mask size. Used to stratify the dataset.
    :param     annotations: (dict list) List of annotations in COCO format (usually from a single image).
    :return    avg_area: (float) Average segmentation/mask size of annotations.
    """

    if len(annotations) > 0:
        areas = 0
        for a in annotations:
            a_seg = np.array(a["segmentation"][0])
            polygon = Polygon(a_seg.reshape(int(len(a_seg) / 2), 2))
            areas += polygon.area
        avg_seg = areas / len(annotations)
    else:
        avg_seg = 0
    return avg_seg

def seed_stats(dataset, stratify=True, output_path="", file_name=""):
    """
    Calculate image-by-image statistics given a COCO dataset.
    :param      dataset: (dict) COCO dataset.
    :return     stats: (pandas.DataFrame) Image-by-image statistics.
    """

    file_names = []
    viable = []
    nonviable = []
    empty = []
    total_counts = []
    viability_ratios = []
    avg_segm_sizes = []
    avg_bbox_sizes = []
    for image in dataset["images"]:
        annotations = extract_annotations(dataset["annotations"], image["id"])
        file_names.append(image["file_name"])
        viable.append(len([a for a in annotations if a["category_id"] == 1]))
        nonviable.append(len([a for a in annotations if a["category_id"] == 2]))
        empty.append(len([a for a in annotations if a["category_id"] == 3]))
        total_counts.append(len(annotations))
        viability_ratios.append(len([a for a in annotations if a["category_id"] == 1]) / len(annotations) if len(annotations) > 0 else 0)
        avg_segm_sizes.append(avg_seg_size(annotations))
        avg_bbox_sizes.append(avg_bbox_size(annotations))

    stats = pd.DataFrame(
        {
            "file_names": file_names,
            "viable": viable,
            "nonviable": nonviable,
            "empty": empty,
            "total": total_counts,
            "viability_ratio": viability_ratios,
            "avg_segm_sizes": np.sqrt(avg_segm_sizes),
            "avg_bbox_sizes": np.sqrt(avg_bbox_sizes)
         }
    )
    summary = None
    if stratify:
        total_bin = pd.qcut(total_counts, q=4, labels=False, duplicates='drop')
        viable_bin = pd.qcut(viable, q=3, labels=False, duplicates='drop')
        size_bin = pd.qcut(avg_segm_sizes, q=4, labels=False, duplicates='drop')
        stratification_stats = pd.DataFrame(
            {
                "total_bin": total_bin,
                "viable_bin": viable_bin,
                "size_bin": size_bin
            }
        )
        stats = pd.concat([stats, stratification_stats], axis=1)
    else:
        summary = stats.describe(include="all")
        whole_dataset = pd.Series({
            "file_names": "whole_dataset",
            "viable": sum(viable),
            "nonviable": sum(nonviable),
            "empty": sum(empty),
            "total": sum(total_counts),
            "viability_ratio": np.mean(viability_ratios),
            "avg_segm_sizes": np.mean(np.sqrt(avg_segm_sizes)),
            "avg_bbox_sizes": np.mean(np.sqrt(avg_bbox_sizes))
        })
        stats.loc[len(stats)] = whole_dataset

    # # Creates graph to visualise seed count histogram.
    # plt.hist(stats["total"], bins=20, edgecolor='black')
    # plt.xlabel('Total seed count')
    # plt.ylabel('Number of images')
    # plt.show()

    if output_path:
        if not file_name:
            file_name = "seed_stats"
        stats.to_csv(os.path.join(output_path, file_name + ".csv"))
        if summary is not None:
            summary.to_csv(os.path.join(output_path, file_name + "_summary.csv"))

    return stats


def stratify(stats, test_ratio=0.2, val_ratio=0.2):
    """
    Stratify the dataset using specific numerical values. Currently, uses the number of viable seeds and the average seed size in each image.
    :param      stats: (pandas.DataFrame) Image-by-image statistics of the dataset.
    :param      test_ratio: (float) Ratio of the dataset to split into the test set used for calculating final model performance.
    :param      val_ratio: (float) Ratio of the dataset to split into the validation set used for evaluating performance during training.
    :return     train: (pandas.DataFrame) Statistics of the images stratified into the training dataset.
    :return     test: (pandas.DataFrame) Statistics of the images stratified into the testing dataset.
    :return     val: (pandas.DataFrame) Statistics of the images stratified into the validation dataset.
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
def train_test_split_coco(path_to_annotations, val_ratio=0.2, test_ratio=0.2,
                          train_imgs=[], val_imgs=[], test_imgs=[], random_seed=None):

    if random_seed != None:
        random.seed(random_seed)
    with open(path_to_annotations, 'r') as file:
        all = json.load(file)

    # List all file names
    all_imgs = []
    for img in all["images"]:
        all_imgs.append(img["file_name"])

    # Randomly select test and validation images if no lists were given
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

    # Remove images selected for testing and validation if they were randomised
    if len(train_imgs) == 0:
        train_imgs = copy.deepcopy(all_imgs)
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

    # Remove the appropriate annotations from the json file
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
    :param      dataset_path: (string) Directory of the COCO dataset.
    :return:
    """
    with open(dataset_path, 'r') as file:
        data = json.load(file)
    for ann in data["annotations"]:
        ann["iscrowd"] = 0
    with open(dataset_path, 'w') as file:
        json.dump(data, file)

def combine_train_and_val(train_path, val_path, output_path):
    def partition(array, low, high):
        pivot = array[high]
        i = low - 1
        for j in range(low, high):
            if array[j]["id"] <= pivot["id"]:
                i = i + 1
                (array[i], array[j]) = (array[j], array[i])
        (array[i + 1], array[high]) = (array[high], array[i + 1])
        return i + 1

    def quickSort(array, low, high):
        if low < high:
            pi = partition(array, low, high)
            quickSort(array, low, pi - 1)
            quickSort(array, pi + 1, high)

    train_and_val = [train_path, val_path]

    images = []
    for a in train_and_val:
        with open(a, 'r') as file:
            images.extend(json.load(file)["images"])
    quickSort(images, 0, len(images) - 1)
    print(len([i["id"] for i in images]))

    annotations = []
    for a in train_and_val:
        with open(a, 'r') as file:
            annotations.extend(json.load(file)["annotations"])
    import sys
    sys.setrecursionlimit(len(annotations))
    quickSort(annotations, 0, len(annotations) - 1)
    print(len([i["id"] for i in annotations]))

    combined = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {"contributor": "", "date_created": "",
                 "description": "", "url": "", "version": "", "year": ""},
        "categories": [{"id": 1, "name": "Viable", "supercategory": ""},
                       {"id": 2, "name": "Non-Viable", "supercategory": ""},
                       {"id": 3, "name": "Empty", "supercategory": ""}],
        "images": images,
        "annotations": annotations
    }

    with open(output_path, "w") as outfile:
        json.dump(combined, outfile)

def main():
    # Reorganise directories and datasets

    # Currently final data is in T drive, but it may be moved at some point
    # OrchidAnnotationProject_COCO/instances_default.json gives 'TZSegmenting/datasets/dataset1/coco_format/post_quality_check/raw_data/all_post_qc_data.json'
    # OrchidAnnotationProject_COCO/images(?) gives TZSegmenting/datasets/dataset1/all_images
    # CVAT_EXPORTS/pre_QA_annotations give some of the pre quality checked annotations but there may be issues with the duplicates.

    print("Combining pre-quality checked datasets")

    # Merge pre-quality checked base dataset
    datasets_to_merge = [
        "../datasets/dataset1/coco_format/pre_quality_check/raw_data/base_dataset/base_part_1.json",
        "../datasets/dataset1/coco_format/pre_quality_check/raw_data/base_dataset/base_part_2.json",
        "../datasets/dataset1/coco_format/pre_quality_check/raw_data/base_dataset/base_part_3.json",
        "../datasets/dataset1/coco_format/pre_quality_check/raw_data/base_dataset/base_part_4.json",
        "../datasets/dataset1/coco_format/pre_quality_check/raw_data/base_dataset/base_part_5.json"
    ]
    output_path = "../datasets/dataset1/coco_format/pre_quality_check/base_dataset.json"
    combine_preqc_datasets(datasets_to_merge, output_path)
    convert_iscrowd(output_path)

    # ...and corresponding checks (duplicate datasets used for agreement analysis)
    datasets_to_merge = [
        "../datasets/dataset1/coco_format/pre_quality_check/raw_data/analysis_duplicate_1/duplicate_1_part_1.json",
        "../datasets/dataset1/coco_format/pre_quality_check/raw_data/analysis_duplicate_1/duplicate_1_part_2.json",
        "../datasets/dataset1/coco_format/pre_quality_check/raw_data/analysis_duplicate_1/duplicate_1_part_3.json",
        "../datasets/dataset1/coco_format/pre_quality_check/raw_data/analysis_duplicate_1/duplicate_1_part_4.json",
        "../datasets/dataset1/coco_format/pre_quality_check/raw_data/analysis_duplicate_1/duplicate_1_part_5.json"
    ]
    output_path = "../datasets/dataset1/coco_format/pre_quality_check/analysis_duplicate_1.json"
    combine_preqc_datasets(datasets_to_merge, output_path)
    convert_iscrowd(output_path)

    datasets_to_merge = [
        "../datasets/dataset1/coco_format/pre_quality_check/raw_data/analysis_duplicate_2/duplicate_2_part_1.json",
        "../datasets/dataset1/coco_format/pre_quality_check/raw_data/analysis_duplicate_2/duplicate_2_part_2.json",
        "../datasets/dataset1/coco_format/pre_quality_check/raw_data/analysis_duplicate_2/duplicate_2_part_3.json",
        "../datasets/dataset1/coco_format/pre_quality_check/raw_data/analysis_duplicate_2/duplicate_2_part_4.json",
        "../datasets/dataset1/coco_format/pre_quality_check/raw_data/analysis_duplicate_2/duplicate_2_part_5.json"
    ]
    output_path = "../datasets/dataset1/coco_format/pre_quality_check/analysis_duplicate_2.json"
    combine_preqc_datasets(datasets_to_merge, output_path)
    convert_iscrowd(output_path)

    print("Splitting post-quality checked datasets")

    annotations_path = "../datasets/dataset1/coco_format/post_quality_check/raw_data/all_post_qc_data.json"
    output_path = "../datasets/dataset1/coco_format/post_quality_check"
    convert_iscrowd(annotations_path)
    split_postqc_dataset(annotations_path, output_path)

    # ----------------------------------------------------------------------------

    # Calculate seed stats, stratify accordingly

    dataset_path = [
        "../datasets/dataset1/coco_format/pre_quality_check/base_dataset.json",
        "../datasets/dataset1/coco_format/post_quality_check/base_dataset.json"
    ]

    # Use the post quality checked dataset for seed stats and stratification
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

    # ----------------------------------------------------------------------------

    # Split the pre and qpost quality checked datasets using the stratified output, create train, test, and val datasets for COCO and YOLO

    coco_dirs = [
        "../datasets/dataset1/coco_format/pre_quality_check/model_data",
        "../datasets/dataset1/coco_format/post_quality_check/model_data"
    ]

    yolo_dirs = [
        "../datasets/dataset1/yolo_format/pre_quality_check",
        "../datasets/dataset1/yolo_format/post_quality_check"
    ]

    if not os.path.exists("../datasets/dataset1/yolo_format"):
        os.makedirs("../datasets/dataset1/yolo_format")

    for i in range(len(coco_dirs)):
        if not os.path.exists(coco_dirs[i]):
            os.makedirs(coco_dirs[i])
        print("Splitting dataset: " + coco_dirs[i])
        # Splitting the data
        train_set, test_set, val_set = train_test_split_coco(
            dataset_path[i],
            train_imgs=list(train["file_names"].to_dict().values()),
            test_imgs=list(test["file_names"].to_dict().values()),
            val_imgs=list(val["file_names"].to_dict().values()),
            random_seed=42
        )

        # Save the COCO datasets as json files
        with open(os.path.join(coco_dirs[i], "train.json"), "w") as outfile:
            json.dump(train_set, outfile)
        with open(os.path.join(coco_dirs[i], "test.json"), "w") as outfile:
            json.dump(test_set, outfile)
        with open(os.path.join(coco_dirs[i], "val.json"), "w") as outfile:
            json.dump(val_set, outfile)

        # Convert the COCO datasets to YOLO
        print("Converting " + coco_dirs[i] + " to YOLO")
        convert_coco(coco_dirs[i], yolo_dirs[i], use_segments=True, cls91to80=False)

        # Create directories for YOLO datasets
        for dir in os.listdir(os.path.join(yolo_dirs[i], "labels")):
            if not os.path.exists(os.path.join(yolo_dirs[i], "images", dir)):
                os.makedirs(os.path.join(yolo_dirs[i], "images", dir))
            for txt in os.listdir(os.path.join(yolo_dirs[i], "labels", dir)):
                img = txt.split(".txt")[0] + ".jpg"
                shutil.copy(os.path.join("../datasets/dataset1/all_images", img),
                            os.path.join(yolo_dirs[i], "images", dir, img))

        # Create yaml files and save to respective directories
        yolo_yaml = {
            "names": {
                0: "Viable",
                1: "Non-Viable",
                2: "Empty"
            },
            "path": yolo_dirs[i],
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
        }
        with open(os.path.join(yolo_dirs[i], 'data.yaml'), 'w') as file:
            yaml.dump(yolo_yaml, file)

    # ----------------------------------------------------------------------------

    # Calculate seed stats for every COCO dataset

    # Create stats directories
    stats_path = "../datasets/dataset1/seed_stats"
    quality_check = ["pre_quality_check", "post_quality_check"]
    if not os.path.exists(stats_path):
        os.makedirs(stats_path)
    for qc in quality_check:
        if not os.path.exists(os.path.join(stats_path, qc)):
            os.makedirs(os.path.join(stats_path, qc))

    coco_dir = "../datasets/dataset1/coco_format"
    for qc in quality_check:
        coco_qc_dir = os.path.join(coco_dir, qc)

        # Calculate stats for pre- and post-quality checked base_datasets and their duplicates
        for dir in os.listdir(coco_qc_dir):
            if dir.endswith(".json"):
                with open(os.path.join(coco_qc_dir, dir), 'r') as file:
                    dataset = json.load(file)
                seed_stats(dataset, stratify=False,
                           output_path=os.path.join(stats_path, qc), file_name=dir.strip(".json") + "_stats")
            else:

                # Calculate stats for training, testing, and validation datasets
                if dir == "model_data":
                    if not os.path.exists(os.path.join(stats_path, qc, dir)):
                        os.makedirs(os.path.join(stats_path, qc, dir))
                    for data in os.listdir(os.path.join(coco_qc_dir, dir)):
                        with open(os.path.join(coco_qc_dir, dir, data), 'r') as file:
                            dataset = json.load(file)
                        seed_stats(dataset, stratify=False,
                                   output_path=os.path.join(stats_path, qc, dir),
                                   file_name=data.strip(".json") + "_stats")

        # Combine post-quality checked training and validation data for final model training
        combine_train_and_val(
            "../datasets/dataset1/coco_format/post_quality_check/model_data/train.json",
            "../datasets/dataset1/coco_format/post_quality_check/model_data/val.json",
            "../datasets/dataset1/coco_format/post_quality_check/model_data/train_and_val.json"
        )

    print("Complete")

if __name__ == "__main__":
    main()