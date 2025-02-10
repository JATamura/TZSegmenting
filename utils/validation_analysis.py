import os
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import Polygon
import krippendorff
import copy
import pandas as pd
from statsmodels.stats import inter_rater as irr
from coco import extract_annotations, rle_to_coco
from sklearn.metrics import confusion_matrix
import seaborn as sn
import itertools

def count_seed_labels_coco(data):
    label_num = {}
    imgs_with_seed = {}
    for label in data["annotations"]:
        label_num[label["category_id"]] = label_num.get(label["category_id"], 0) + 1
        if label["category_id"] == 1:
            img_name = ""
            for i in range(len(data["images"])):
                if data["images"][i]["id"] == label["image_id"]:
                    img_name = data["images"][i]["file_name"]
            imgs_with_seed[img_name] = imgs_with_seed.get(img_name, 0) + 1
    return label_num, imgs_with_seed

def count_seed_labels_yolo(data_path):
    label_num = {}
    imgs_with_seed = {}
    for text_file in os.listdir(data_path):
        text = open(os.path.join(data_path, text_file), "r")
        text = text.read().split("\n")
        for annotation in text:
            if annotation != '':
                label_num[annotation[0]] = label_num.get(annotation[0], 0) + 1
                if int(annotation[0]) == 0:
                    imgs_with_seed[text_file] = imgs_with_seed.get(text_file, 0) + 1
    return label_num, imgs_with_seed

def find_val_images(labels_path):
    with open(labels_path, 'r') as file:
        data = json.load(file)
    val_images = set()
    for image in data["images"]:
        if "_" in image["file_name"]:
            val_images.add(image["file_name"][:3])

    val_image_names = []
    for image in sorted(val_images):
        image_names = [image, image + "_1", image + "_2"]
        val_image_names.append(image_names)

    val_image_annotations = []
    for image_set in val_image_names:
        missing_images = []
        image_annotations = []
        for image in image_set:
            a = extract_annotations(data, image)
            print(image + ": " + str(len(a)))
            if len(a) == 0:
                missing_images.append(image)
            else:
                image_annotations.append(a)
        print("---------")
        if len(image_annotations) == 3:
            set_of_3 = {}
            for i in range(3):
                set_of_3[image_set[i]] = image_annotations[i]
            val_image_annotations.append(set_of_3)
    return val_image_annotations

def validate_2(a1, a2, iou_threshold=0.5, image_name="", verbose=True):
    annotations_1 = copy.deepcopy(a1)
    annotations_2 = copy.deepcopy(a2)
    if verbose:
        print(image_name + " Total annotations")
        print("A1: " + str(len(annotations_1)))
        print("A2: " + str(len(annotations_2)))
    same_label = []
    diff_label = []
    undetected = []
    for a_1 in annotations_1:
        if type(a_1["segmentation"]) == list:
            a_1_seg = np.array(a_1["segmentation"][0])
        else:
            a_1_seg = np.array(a_1["segmentation_coords"])
        polygon_1 = Polygon(a_1_seg.reshape(int(len(a_1_seg)/2), 2))
        max_overlap = 0
        max_a_2 = {}
        for a_2 in annotations_2:
            if type(a_2["segmentation"]) == list:
                a_2_seg = np.array(a_2["segmentation"][0])
            else:
                a_2_seg = np.array(a_2["segmentation_coords"])
            polygon_2 = Polygon(a_2_seg.reshape(int(len(a_2_seg)/2), 2))
            iou = 0
            if polygon_1.is_valid and polygon_2.is_valid:
                intersect = polygon_1.intersection(polygon_2).area
                union = polygon_1.union(polygon_2).area
                iou = intersect / union
            if iou > max_overlap:
                max_overlap = iou
                max_a_2 = a_2
        if max_overlap > iou_threshold:
            if a_1["category_id"] == max_a_2["category_id"]:
                same_label.append({"a_1":a_1, "a_2":max_a_2})
            else:
                diff_label.append({"a_1":a_1, "a_2":max_a_2})
            annotations_2.remove(max_a_2)
        else:
            undetected.append({"a_1":a_1, "a_2":None})
    for a_2 in annotations_2:
        undetected.append({"a_1": None, "a_2": a_2})
    if verbose:
        print("---------------")
        print("Same labels: " + str(len(same_label)))
        print("Different labels: " + str(len(diff_label)))
        print("Undetected: " + str(len(undetected)))
        print("---------------")
    return same_label, diff_label, undetected

def validate_3(a1, a2, a3, iou_threshold=0.5, image_name="", verbose=True):
    annotations_1 = copy.deepcopy(a1)
    annotations_2 = copy.deepcopy(a2)
    annotations_3 = copy.deepcopy(a3)
    if verbose:
        print(image_name + " Total annotations")
        print("A1: " + str(len(annotations_1)))
        print("A2: " + str(len(annotations_2)))
        print("A3: " + str(len(annotations_3)))
    same_label = []
    diff_label = []
    undetected = []
    nv = 0
    for annotations in [annotations_1, annotations_2, annotations_3]:
        for a in annotations:
            if type(a["segmentation"]) == list:
                a_seg = np.array(a["segmentation"][0])
            else:
                a_seg = np.array(a["segmentation_coords"])
            polygon = Polygon(a_seg.reshape(int(len(a_seg)/2), 2))
            if not polygon.is_valid:
                nv += 1
    for a_1 in annotations_1:
        if type(a_1["segmentation"]) == list:
            a_1_seg = np.array(a_1["segmentation"][0])
        else:
            a_1_seg = np.array(a_1["segmentation_coords"])
        polygon_1 = Polygon(a_1_seg.reshape(int(len(a_1_seg)/2), 2))
        max_overlap_2 = 0
        max_a_2 = {}
        max_polygon_2 = None
        for a_2 in annotations_2:
            if type(a_2["segmentation"]) == list:
                a_2_seg = np.array(a_2["segmentation"][0])
            else:
                a_2_seg = np.array(a_2["segmentation_coords"])
            polygon_2 = Polygon(a_2_seg.reshape(int(len(a_2_seg)/2), 2))
            iou = 0
            if polygon_1.is_valid and polygon_2.is_valid:
                intersect = polygon_1.intersection(polygon_2).area
                union = polygon_1.union(polygon_2).area
                iou = intersect / union
            if iou > max_overlap_2:
                max_overlap_2 = iou
                max_a_2 = a_2
                max_polygon_2 = polygon_2
        max_overlap_3 = 0
        max_overlap_all = 0
        max_a_3 = {}
        for a_3 in annotations_3:
            if type(a_3["segmentation"]) == list:
                a_3_seg = np.array(a_3["segmentation"][0])
            else:
                a_3_seg = np.array(a_3["segmentation_coords"])
            polygon_3 = Polygon(a_3_seg.reshape(int(len(a_3_seg) / 2), 2))
            iou = 0
            if polygon_1.is_valid and polygon_3.is_valid:
                intersect = polygon_1.intersection(polygon_3).area
                union = polygon_1.union(polygon_3).area
                iou = intersect / union
            if iou > max_overlap_3:
                max_overlap_3 = iou
                max_a_3 = a_3
            if max_overlap_2 > iou_threshold and max_overlap_3 > iou_threshold:
                iou = 0
                if max_polygon_2.is_valid and polygon_3.is_valid:
                    intersect = max_polygon_2.intersection(polygon_3).area
                    union = max_polygon_2.union(polygon_3).area
                    iou = intersect / union
                if iou > max_overlap_all:
                    max_overlap_all = iou
        if max_overlap_all > iou_threshold:
            if a_1["category_id"] == max_a_2["category_id"] == max_a_3["category_id"]:
                same_label.append({"a_1":a_1, "a_2":max_a_2, "a_3":max_a_3})
            else:
                diff_label.append({"a_1":a_1, "a_2":max_a_2, "a_3":max_a_3})
            annotations_2.remove(max_a_2)
            annotations_3.remove(max_a_3)
        else:
            if max_overlap_2 > iou_threshold:
                undetected.append({"a_1":a_1, "a_2":max_a_2, "a_3": None})
                annotations_2.remove(max_a_2)
            elif max_overlap_3 > iou_threshold:
                undetected.append({"a_1": a_1, "a_2": None, "a_3": max_a_3})
                annotations_3.remove(max_a_3)
            else:
                undetected.append({"a_1":a_1, "a_2": None, "a_3": None})
    for a_2 in annotations_2:
        if type(a_2["segmentation"]) == list:
            a_2_seg = np.array(a_2["segmentation"][0])
        else:
            a_2_seg = np.array(a_2["segmentation_coords"])
        polygon_2 = Polygon(a_2_seg.reshape(int(len(a_2_seg) / 2), 2))
        max_overlap_3 = 0
        max_a_3 = {}
        for a_3 in annotations_3:
            if type(a_3["segmentation"]) == list:
                a_3_seg = np.array(a_3["segmentation"][0])
            else:
                a_3_seg = np.array(a_3["segmentation_coords"])
            polygon_3 = Polygon(a_3_seg.reshape(int(len(a_3_seg) / 2), 2))
            iou = 0
            if polygon_2.is_valid and polygon_3.is_valid:
                intersect = polygon_2.intersection(polygon_3).area
                union = polygon_2.union(polygon_3).area
                iou = intersect / union
            if iou > max_overlap_3:
                max_overlap_3 = iou
                max_a_3 = a_3
        if max_overlap_3 > iou_threshold:
            undetected.append({"a_1": None, "a_2": a_2, "a_3": max_a_3})
            annotations_3.remove(max_a_3)
        else:
            undetected.append({"a_1": None, "a_2": a_2, "a_3": None})
    for a_3 in annotations_3:
        undetected.append({"a_1": None, "a_2": None, "a_3": a_3})
    if verbose:
        print("---------------")
        print("Same labels: " + str(len(same_label)))
        print("Different labels: " + str(len(diff_label)))
        print("Undetected: " + str(len(undetected)))
        print("-----------------")
    return same_label, diff_label, undetected

def plot_segmentations(a_1, a_2=None, a_3=None, ax=None, img=None):
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    if not a_1 is None:
        if type(a_1["segmentation"]) == list:
            a_1_seg = np.array(a_1["segmentation"][0])
        else:
            a_1 = rle_to_coco(a_1)[0]
            a_1_seg = np.array(a_1["segmentation_coords"])
        p1 = np.array(a_1_seg).reshape(int(len(a_1_seg)/2), 2)
        poly1 = Polygon(p1)
        if a_2 is None and a_3 is None:
            if a_1["category_id"] == 2:
                ax.plot(*poly1.exterior.xy, c='green')
            elif a_1["category_id"] == 3:
                ax.plot(*poly1.exterior.xy, c='red')
            elif a_1["category_id"] == 4:
                ax.plot(*poly1.exterior.xy, c='black')
        else:
            ax.plot(*poly1.exterior.xy, c='red')
    if not a_2 is None:
        if type(a_2["segmentation"]) == list:
            a_2_seg = np.array(a_2["segmentation"][0])
        else:
            a_2 = rle_to_coco(a_2)[0]
            a_2_seg = np.array(a_2["segmentation_coords"])
        p2 = np.array(a_2_seg).reshape(int(len(a_2_seg)/2), 2)
        poly2 = Polygon(p2)
        ax.plot(*poly2.exterior.xy, c='green')
    if not a_3 is None:
        if type(a_3["segmentation"]) == list:
            a_3_seg = np.array(a_3["segmentation"][0])
        else:
            a_3 = rle_to_coco(a_3)[0]
            a_3_seg = np.array(a_3["segmentation_coords"])
        p3 = np.array(a_3_seg).reshape(int(len(a_3_seg)/2), 2)
        poly3 = Polygon(p3)
        ax.plot(*poly3.exterior.xy, c='blue')
    ax.set_xlim([0, 2560])
    ax.set_ylim([2048, 0])
    if a_2 is None and a_3 is None:
        custom_lines = [Line2D([0], [0], color='green', lw=2, label='Viable'),
                        Line2D([0], [0], color='red', lw=2, label='Non-Viable'),
                        Line2D([0], [0], color='black', lw=2, label='Empty')]
    else:
        custom_lines = [Line2D([0], [0], color='blue', lw=2, label='0'),
                        Line2D([0], [0], color='red', lw=2, label='1'),
                        Line2D([0], [0], color='green', lw=2, label='2')]
    # ax.legend(handles=custom_lines)
    if not img is None:
        ax.imshow(img)
    if show:
        plt.show()

def compute_krippendorff(annotations, count_undetected=True):
    all_a = annotations[0] + annotations[1] + annotations[2]
    annotators = [[] for i in range(3)]
    for a in all_a:
        if count_undetected:
            for i in range(3):
                if a["a_" + str(i+1)] is not None:
                    annotators[i].append(a["a_" + str(i+1)]["category_id"])
                else:
                    annotators[i].append(0)
        else:
            if None not in a.values():
                for i in range(3):
                    if a["a_" + str(i + 1)] is not None:
                        annotators[i].append(a["a_" + str(i + 1)]["category_id"])
    if annotators[0] == annotators[1] == annotators[2]:
        return annotators, 1
    else:
        return annotators, krippendorff.alpha(annotators, level_of_measurement="nominal")

def compute_fleiss(annotations, count_undetected=True, method='uniform'):
    all_a = annotations[0] + annotations[1] + annotations[2]
    annotators = [[] for i in range(3)]
    for a in all_a:
        if count_undetected:
            for i in range(3):
                if a["a_" + str(i + 1)] is not None:
                    annotators[i].append(a["a_" + str(i + 1)]["category_id"])
                else:
                    annotators[i].append(0)
        else:
            if None not in a.values():
                for i in range(3):
                    if a["a_" + str(i + 1)] is not None:
                        annotators[i].append(a["a_" + str(i + 1)]["category_id"])
    if annotators[0] == annotators[1] == annotators[2]:
        return annotators, 1
    else:
        return annotators, irr.fleiss_kappa(irr.aggregate_raters(np.array(annotators).transpose())[0], method=method)

if __name__ == "__main__":
    with open("../datasets/coco/part3/part3_postqa_coco/annotations/instances_default.json", 'r') as file:
        dataset = json.load(file)
    for i in dataset["images"]:
        print("id: ", i["file_name"], i["id"])
        print(len(extract_annotations(dataset["annotations"], i["id"])))

    fig, ax = plt.subplots()
    for a in extract_annotations(dataset["annotations"], 23):
        plot_segmentations(a, ax=ax)
    im = cv2.imread("../datasets/coco/part3/part3_postqa_coco/images/default/283.jpg")
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.show()
    exit()

    # Path to all images
    image_dir = "images/all_parts/original/images/default/"

    # Paths to dataset and corresponding checks
    path_to_part = "images/all_parts/original/annotations/instances_default.json"
    path_to_check_1 = "images/all_parts/check1/annotations/instances_default.json"
    path_to_check_2 = "images/all_parts/check2/annotations/instances_default.json"

    with open(path_to_part, 'r') as file:
        part_1 = json.load(file)
    for a in range(len(part_1["annotations"])):
        if type(part_1["annotations"][a]["segmentation"]) != list:
            part_1["annotations"][a] = rle_to_coco(part_1["annotations"][a])[0]

    with open(path_to_check_1, 'r') as file:
        check_1 = json.load(file)
    for a in range(len(check_1["annotations"])):
        if type(check_1["annotations"][a]["segmentation"]) != list:
            check_1["annotations"][a] = rle_to_coco(check_1["annotations"][a])[0]

    with open(path_to_check_2, 'r') as file:
        check_2 = json.load(file)
    for a in range(len(check_2["annotations"])):
        if type(check_2["annotations"][a]["segmentation"]) != list:
            check_2["annotations"][a] = rle_to_coco(check_2["annotations"][a])[0]

    val_image_names = []
    if check_1["images"] == check_2["images"]:
        print("All validation images have the same name")
        val_image_names = pd.DataFrame(check_1["images"]).loc[:, "file_name"]
    else:
        print("Check 1 =/= Check 2")

    all_annotations = [part_1, check_1, check_2]
    val_image_annotations = {}
    for v in val_image_names:
        set_of_3 = []
        for annotations in all_annotations:
            a = extract_annotations(annotations, v[:3])
            set_of_3.append(a)
        val_image_annotations[v] = set_of_3

    ious = [0.5]
    for i in ious:
        iou_thresh = i
        print(iou_thresh)
        agreement = {}
        all_seed_validations = [[], [], []]
        for v in val_image_names:
            validation = validate_3(val_image_annotations[v][0],
                                            val_image_annotations[v][1],
                                            val_image_annotations[v][2],
                                            iou_thresh, v, False)
            for i in range(3):
                all_seed_validations[i] += validation[i]
            stats = {"same_label": len(validation[0]), "different_label": len(validation[1]), "not_labeled": len(validation[2])}
            kd = compute_krippendorff(validation, count_undetected=True)
            stats["krippendorff_with_undetected"] = kd[1]
            kd = compute_krippendorff(validation, count_undetected=False)
            stats["krippendorff_without_undetected"] = kd[1]
            un = compute_fleiss(validation, count_undetected=True)
            stats["uniform_with_undetected"] = un[1]
            un = compute_fleiss(validation, count_undetected=False)
            stats["uniform_without_undetected"] = un[1]
            agreement[v] = stats
        agreement = pd.DataFrame(agreement)
        agreement['img_mean'] = agreement.mean(axis=1)
        agreement['seed_mean'] = [len(all_seed_validations[0]),
                              len(all_seed_validations[1]),
                              len(all_seed_validations[2]),
                              compute_krippendorff(all_seed_validations, count_undetected=True)[1],
                              compute_krippendorff(all_seed_validations, count_undetected=False)[1],compute_fleiss(all_seed_validations, count_undetected=True)[1],
                              compute_fleiss(all_seed_validations, count_undetected=False)[1]]
        print(agreement.loc[:, ["img_mean", "seed_mean"]])
        # agreement.to_csv("validation_stats/" + str(iou_thresh) + ".csv", index=True)

    all_annotations = compute_krippendorff(all_seed_validations, count_undetected=True)[0]
    fig,ax = plt.subplots(1, 3)
    sn.heatmap(pd.DataFrame(confusion_matrix(all_annotations[0], all_annotations[1])), annot=True, fmt=".0f", ax=ax[0])
    sn.heatmap(pd.DataFrame(confusion_matrix(all_annotations[0], all_annotations[2])), annot=True, fmt=".0f", ax=ax[1])
    sn.heatmap(pd.DataFrame(confusion_matrix(all_annotations[1], all_annotations[2])), annot=True, fmt=".0f", ax=ax[2])
    plt.show()

    all_a = list(zip(all_annotations[0], all_annotations[1], all_annotations[2]))
    all_a_converted = []
    for set_of_three in all_a:
        set_of_three_converted = []
        for a in set_of_three:
            switch = {
                0: 'Undetected',
                1: 'Seed',
                2: 'Viable',
                3: 'Non-Viable',
                4: 'Empty'
            }
            set_of_three_converted.append(switch.get(a))
        all_a_converted.append(set_of_three_converted)

    pair_agreement = {}
    for set_of_three in all_a_converted:
        all_combinations = list(itertools.combinations(set_of_three, 2))
        for c in all_combinations:
            pair_agreement[tuple(set(c))] = pair_agreement.get(tuple(set(c)), 0) + 1
    pair_agreement["Total"] = sum(pair_agreement.values())
    pair_agreement = pd.DataFrame(pair_agreement.values(), index=list(pair_agreement.keys()), columns=["Number of annotations"])
    print(pair_agreement.sort_values("Number of annotations", ascending=False))

    unique = {}
    for set_of_three in all_a_converted:
        unique[tuple(sorted(set_of_three))] = unique.get(tuple(sorted(set_of_three)), 0) + 1
    unique["Total"] = sum(unique.values())
    unique = pd.DataFrame(unique.values(), index=list(unique.keys()), columns=["Number of annotations"])
    print(unique.sort_values("Number of annotations", ascending=False))