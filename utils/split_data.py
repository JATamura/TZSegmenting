import json
import os
import random
import shutil
import copy
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
import pandas as pd
from coco import extract_annotations, rle_to_coco
from sklearn.model_selection import train_test_split
from ultralytics.data.converter import convert_coco
import yaml

def avg_seed_size(annotations):
    areas = 0
    for a in annotations:
        if type(a["segmentation"]) == list:
            a_seg = np.array(a["segmentation"][0])
        else:
            a = rle_to_coco(a)[0]
            a_seg = np.array(a["segmentation_coords"])
        polygon = Polygon(a_seg.reshape(int(len(a_seg) / 2), 2))
        areas += polygon.area
    return areas/len(annotations)

def stratify(dataset, test_ratio=0.2, val_ratio=0.2):
    file_names = []
    viable = []
    nonviable = []
    empty = []
    total_counts = []
    viable_ratios = []
    avg_sizes = []
    for image in dataset["images"]:
        annotations = extract_annotations(dataset["annotations"], image["id"])
        if len(annotations) > 0:
            file_names.append(image["file_name"])
            viable.append(len([a for a in annotations if a["category_id"] == 1]))
            nonviable.append(len([a for a in annotations if a["category_id"] == 2]))
            empty.append(len([a for a in annotations if a["category_id"] == 3]))
            total_counts.append(len(annotations))
            viable_ratios.append(len([a for a in annotations if a["category_id"] == 1]) / len(annotations))
            avg_sizes.append(avg_seed_size(annotations))

    total_bin = pd.qcut(total_counts, q=4, labels=False, duplicates='drop')
    viable_bin = pd.qcut(viable, q=3, labels=False, duplicates='drop')
    size_bin = pd.qcut(avg_sizes, q=4, labels=False, duplicates='drop')

    stats = pd.DataFrame(
        {"file_names": file_names,
         "viable": viable,
         "nonviable": nonviable,
         "empty": empty,
         "total": total_counts,
         "avg_sizes": np.sqrt(avg_sizes),
         "total_bin": total_bin,
         "viable_bin": viable_bin,
         "size_bin": size_bin}
    )

    stats["stratify"] = (
        # stats["total_bin"].astype(str) + "_" +
        stats["viable_bin"].astype(str) + "_" +
        stats["size_bin"].astype(str)
    )

    print(pd.cut(stats["total"], [i for i in range(0, 1200, 50)]).value_counts(sort=False))

    stats = stats[stats.total < 800]
    print(stats.shape)
    print(stats.to_string())

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
        annotations_all = json.load(file)

    # list all file names
    all_imgs = []
    for img in annotations_all["images"]:
        all_imgs.append(img["file_name"])
    train_imgs = copy.deepcopy(all_imgs)

    # randomly select validation and test images if no lists were given
    if len(val_imgs) == 0:
        for i in range(int(len(all_imgs) * val_ratio)):
            img = random.choice(train_imgs)
            while img in val_imgs:
                img = random.choice(train_imgs)
            val_imgs.append(img)
    if len(test_imgs) == 0:
        for i in range(int(len(all_imgs) * test_ratio)):
            img = random.choice(train_imgs)
            while img in test_imgs:
                img = random.choice(train_imgs)
            test_imgs.append(img)

    for img in val_imgs + test_imgs:
        train_imgs.remove(img)

    # remove appropriate images from the json file
    imgs_with_issues = []
    for img in annotations_all["images"]:
        annotations = extract_annotations(annotations_all["annotations"], img["id"])
        if len(annotations) > 300:
            imgs_with_issues.append((img["file_name"], len(annotations), img))
        if len(annotations) < 1:
            imgs_with_issues.append((img["file_name"], len(annotations), img))
        if img["file_name"] == "425.jpg":
            imgs_with_issues.append((img["file_name"], len(annotations), img))


    for file_name, object_count, img in imgs_with_issues:
        print(file_name + " was removed for having " + str(object_count) + " objects")
        annotations_all["images"].remove(img)

    print(len(annotations_all["images"]))
    exit()

    annotations_train = copy.deepcopy(annotations_all)
    annotations_val = copy.deepcopy(annotations_all)
    annotations_test = copy.deepcopy(annotations_all)

    for img in annotations_all["images"]:
        if not img["file_name"] in train_imgs:
            annotations_train["images"].remove(img)
        if not img["file_name"] in val_imgs:
            annotations_val["images"].remove(img)
        if not img["file_name"] in test_imgs:
            annotations_test["images"].remove(img)

    # remove the appropriate annotations from the json file
    train_ids = [img["id"] for img in annotations_train["images"]]
    val_ids = [img["id"] for img in annotations_val["images"]]
    test_ids = [img["id"] for img in annotations_test["images"]]
    for a in annotations_all["annotations"]:
        if not a["image_id"] in train_ids:
            annotations_train["annotations"].remove(a)
        if not a["image_id"] in val_ids:
            annotations_val["annotations"].remove(a)
        if not a["image_id"] in test_ids:
            annotations_test["annotations"].remove(a)

    # save the json files
    with open(os.path.join(output_path, "train.json"), "w") as outfile:
        json.dump(annotations_train, outfile)
    with open(os.path.join(output_path, "val.json"), "w") as outfile:
        json.dump(annotations_val, outfile)
    with open(os.path.join(output_path, "test.json"), "w") as outfile:
        json.dump(annotations_test, outfile)

if __name__ == "__main__":
    coco_dir = ["../datasets/dataset1/coco/pre_quality_check/all_preqc.json",
                "../datasets/dataset1/coco/post_quality_check/all_postqc.json"]
    output_path = ["../datasets/dataset1/coco/preqc_model_data",
                   "../datasets/dataset1/coco/postqc_model_data"]
    yolo_dir = ["../datasets/dataset1/yolo/pre_qc",
                "../datasets/dataset1/yolo/post_qc"]

    print("Loading full dataset")
    with open(coco_dir[1], 'r') as file:
        dataset = json.load(file)

    print("Stratifying data")
    train, test, val = stratify(dataset)
    print("train size: " + str(len(train)))
    print("test size: " + str(len(test)))
    print("val size: " + str(len(val)))

    print("Finished stratification")
    # exit()

    # for i in range(2):
    for i in [1]:
        print("Splitting dataset " + str(i))
        # # Splitting the data
        train_test_split_coco(coco_dir[i], output_path[i],
                              val_imgs=list(val["file_names"].to_dict().values()),
                              test_imgs=list(test["file_names"].to_dict().values()),
                              random_seed=42)

        print("Converting dataset " + str(i) + " to YOLO")
        convert_coco(output_path[i], yolo_dir[i], use_segments=True, cls91to80=False)

        for dir in os.listdir(os.path.join(yolo_dir[i], "labels")):
            if not os.path.exists(os.path.join(yolo_dir[i], "images", dir)):
                os.makedirs(os.path.join(yolo_dir[i], "images", dir), exist_ok=True)
            for txt in os.listdir(os.path.join(yolo_dir[i], "labels", dir)):
                img = txt.split(".txt")[0] + ".jpg"
                shutil.copy(os.path.join("../datasets/dataset1/all_images", img),
                            os.path.join(yolo_dir[i], "images", dir, img))

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

    print("Checking if data is split equally")
    print("Train: " + str(os.listdir("../datasets/dataset1/yolo/pre_qc/labels/train")
          == os.listdir("../datasets/dataset1/yolo/post_qc/labels/train")))
    print("Test: " + str(os.listdir("../datasets/dataset1/yolo/pre_qc/labels/test")
          == os.listdir("../datasets/dataset1/yolo/post_qc/labels/test")))
    print("Val: " + str(os.listdir("../datasets/dataset1/yolo/pre_qc/labels/val")
          == os.listdir("../datasets/dataset1/yolo/post_qc/labels/val")))
