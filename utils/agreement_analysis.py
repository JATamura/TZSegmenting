import os
import json
import numpy as np
import shapely
from IPython.core.display_functions import display
from shapely.geometry import Polygon
import krippendorff
import copy
import pandas as pd
from statsmodels.stats import inter_rater as irr
from configure_data import extract_annotations
import itertools


def count_labels(dataset, search_cat=""):
    """
    Counts the number of objects with each category in a COCO dataset. Can also find images with objects of a specific category and count how many objects of that category are in each image.
    :param      dataset: (dict) COCO dataset.
    :param      search_cat: (string) Category name to search for.
    :return:    label_num: (dict) Dictionary with each category that appears in the dataset as the key and number of occurrences as the value.
    :return:    imgs_with_search_cat: (dict) Dictionary with the image_name as the key and number of objects with the search category in said image as the value.
    """
    cat_num = {}
    imgs_with_search_cat = {}
    search_id = next((i for i in dataset["categories"] if i["name"] == search_cat), None)
    for cat in dataset["categories"]:
        cat_num[cat["id"]] = 0
    for annotation in dataset["annotations"]:
        cat_num[annotation["category_id"]] = cat_num.get(annotation["category_id"], 0) + 1
        if search_id and annotation["category_id"] == search_id["id"]:
            img_name = ""
            for i in range(len(dataset["images"])):
                if dataset["images"][i]["id"] == annotation["image_id"]:
                    img_name = dataset["images"][i]["file_name"]
            imgs_with_search_cat[img_name] = imgs_with_search_cat.get(img_name, 0) + 1
    if imgs_with_search_cat:
        return [cat_num, imgs_with_search_cat]
    else:
        return [cat_num]


def annotations_to_polygons(annotation_sets):
    """
    Converts a list of annotations from different COCO datasets to a nested array of dictionaries that hold the annotation and its corresponding polygon.
    :param      annotation_sets: (list) 2D array with each inner list containing annotations from a COCO dataset.
    :return:    ann_to_polygon_sets: (list) 2D array with each inner list containing annotations and their corrsponding polygons.
    """
    ann_to_polygon_sets = [[] for a in annotation_sets]
    for i, annotation_set in enumerate(annotation_sets):
        for annotation in annotation_set:
            ann_to_polygon = {}
            a_seg = np.array(annotation["segmentation"][0])
            polygon = Polygon(np.array(a_seg).reshape(int(len(a_seg) / 2), 2))
            ann_to_polygon["annotation"] = annotation
            ann_to_polygon["polygon"] = polygon
            ann_to_polygon_sets[i].append(ann_to_polygon)
    return ann_to_polygon_sets


def compare_annotations_2(ann_to_polygon_1, ann_to_polygon_2, iou_threshold=0.5, image_name="", verbose=True):
    """
    Given 2 sets of annotations, counts the number of annotations with differing levels of agreement in segmentation and classification (see return values)
    :param      ann_to_polygon_1: (list dict) Annotations made by an annotator and their corresponding polygons. Assumes all segmentations are in COCO formatting and not RLE.
    :param      ann_to_polygon_2: (list dict) Annotations made by a second annotator and their corresponding polygons. Assumes the same formatting as a1.
    :param      iou_threshold: (float) Intersection over union (IoU) threshold. Controls degree of overlap between the masks of 2 annotations for them to be considered as annotating the same object.
    :param      image_name: (string) Name of image to compare annotations for. Useful when outputting result.
    :param      verbose: (bool) Outputs result if True, else just returns lists.
    :return     same_label: (list dict) List of annotations with agreement in segmentation and classification by both annotators.
    :return     different_label: (array dict) List of annotations with agreement in segmentation by both annotators but disagreement in classification.
    :return     undetected: (array dict) List of annotations with disagreement in segmentation and classification. These are usually missed seeds (under-annotations) or debris that was seen as a seed (over-annotations).
    """
    if verbose:
        print("---------------")
        if image_name:
            print(image_name)
        print("Total annotations")
        print("Annotator 1: " + str(len(ann_to_polygon_1)))
        print("Annotator 2: " + str(len(ann_to_polygon_2)))
    same_label = []
    different_label = []
    undetected_label = []

    for a_1 in ann_to_polygon_1:
        max_iou = 0
        max_a_2 = {}
        for a_2 in ann_to_polygon_2:

            # Calculate the intersect over union (IoU) of the two annotations using their polygons
            if a_1["polygon"].is_valid and a_2["polygon"].is_valid:
                intersect = a_1["polygon"].intersection(a_2["polygon"]).area
                union = a_1["polygon"].union(a_2["polygon"]).area
                iou = intersect / union

                # Change the highest IoU score and corresponding annotation accordingly
                if iou > max_iou:
                    max_iou = iou
                    max_a_2 = a_2

        # Check if the highest matching annotation pair has an IoU score higher than the threshold
        if max_iou > iou_threshold:

            # Append to same_label or diff_label according to the annotations' categories
            if a_1["annotation"]["category_id"] == max_a_2["annotation"]["category_id"]:
                same_label.append({"a_1": a_1["annotation"], "a_2": max_a_2["annotation"]})
            else:
                different_label.append({"a_1": a_1["annotation"], "a_2": max_a_2["annotation"]})

            # Remove a_2 from the pool of comparable annotations
            ann_to_polygon_2.remove(max_a_2)

        # If there is no matching pair, append to undetected_label
        else:
            undetected_label.append({"a_1": a_1["annotation"], "a_2": None})

    # Append all remaining annotations in annotations_2 to undetected as they would not have had a matching annotation in annotations_1
    for a_2 in ann_to_polygon_2:
        undetected_label.append({"a_1": None, "a_2": a_2["annotation"]})

    if verbose:
        print("Same labels: " + str(len(same_label)))
        print("Different labels: " + str(len(different_label)))
        print("Undetected: " + str(len(undetected_label)))
        print("---------------")
    return same_label, different_label, undetected_label


def compare_annotations_3(ann_to_polygon_1, ann_to_polygon_2, ann_to_polygon_3, iou_threshold=0.5, image_name="",
                          verbose=True):
    """
    Given 3 sets of annotations, counts the number of annotations with differing levels of agreement in segmentation and classification (see return values)
    :param      ann_to_polygon_1: (list dict) Annotations made by an annotator and their corresponding polygons. Assumes all segmentations are in COCO formatting and not RLE.
    :param      ann_to_polygon_2: (list dict) Annotations made by a second annotator and their corresponding polygons. Assumes the same formatting as a1.
    :param      ann_to_polygon_3: (list dict) Annotations made by a third annotator and their corresponding polygons. Assumes the same formatting as a1.
    :param      iou_threshold: (float) Intersection over union (IoU) threshold. Controls degree of overlap between the masks of 2 annotations for them to be considered as annotating the same object.
    :param      image_name: (string) Name of image to compare annotations for. Useful when outputting result.
    :param      verbose: (bool) Outputs result if True, else just returns lists.
    :return     same_label: (list dict) List of annotations with agreement in segmentation and classification by all annotators.
    :return     different_label: (array dict) List of annotations with agreement in segmentation by all annotators but disagreement in classification by one or more annotators.
    :return     undetected: (array dict) List of annotations with disagreement in segmentation and classification. These are usually missed seeds (under-annotations) or debris that was seen as a seed (over-annotations).
    """
    if verbose:
        print("---------------")
        if image_name:
            print(image_name)
        print("Total annotations")
        print("Annotator 1: " + str(len(ann_to_polygon_1)))
        print("Annotator 2: " + str(len(ann_to_polygon_2)))
        print("Annotator 3: " + str(len(ann_to_polygon_3)))

    same_label = []
    different_label = []
    undetected_label = []

    # Deep copy ann_to_polygon_1 so when removing annotations, the loop doesn't skip items in the list
    for a_1 in copy.deepcopy(ann_to_polygon_1):
        max_iou = 0
        max_a_2 = {}
        max_a_3 = {}
        for a_2 in ann_to_polygon_2:

            # Check if there is any overlap between the first 2 annotations before finding the intersection of all 3 (reduces runtime significantly)
            if a_1["polygon"].is_valid and a_2["polygon"].is_valid:
                test_intersection = a_1["polygon"].intersection(a_2["polygon"]).area

                if test_intersection > 0:
                    for a_3 in ann_to_polygon_3:
                        # Calculate the intersect over union (IoU) for each combination of a_1, a_2, and a_3 using their polygons
                        if a_3["polygon"].is_valid:
                            intersection = shapely.intersection_all(
                                [a_1["polygon"], a_2["polygon"], a_3["polygon"]]).area
                            union = shapely.union_all([a_1["polygon"], a_2["polygon"], a_3["polygon"]]).area
                            iou = intersection / union

                            # Change the highest IoU score and corresponding annotations accordingly
                            if iou > max_iou:
                                max_iou = iou
                                max_a_2 = a_2
                                max_a_3 = a_3
        # Check if the highest matching annotation triplet has an IoU score higher than the threshold
        if max_iou > iou_threshold:

            # Append to same_label or diff_label according to the annotations' categories
            if a_1["annotation"]["category_id"] == max_a_2["annotation"]["category_id"] == max_a_3["annotation"][
                "category_id"]:
                same_label.append(
                    {"a_1": a_1["annotation"], "a_2": max_a_2["annotation"], "a_3": max_a_3["annotation"]})
            else:
                different_label.append(
                    {"a_1": a_1["annotation"], "a_2": max_a_2["annotation"], "a_3": max_a_3["annotation"]})

            # Remove the annotations with unanimous segmentation agreement from the pool of comparable annotations
            ann_to_polygon_1.remove(a_1)
            ann_to_polygon_2.remove(max_a_2)
            ann_to_polygon_3.remove(max_a_3)

    # Extract the annotations from the dictionaries
    annotations_1 = [a["annotation"] for a in ann_to_polygon_1]
    annotations_2 = [a["annotation"] for a in ann_to_polygon_2]
    annotations_3 = [a["annotation"] for a in ann_to_polygon_3]

    # Run compare_annotations_2 for each combination of the 3 remaining sets (annotations_1 and annotations_2 here)
    same, different, undetected = compare_annotations_2(ann_to_polygon_1, ann_to_polygon_2,
                                                        iou_threshold=iou_threshold,
                                                        image_name=image_name + ": a_1, a_2", verbose=verbose)

    # Append the matching annotation pairs to undetected and remove from the remaining pool accordingly
    for annotation_pair in (same + different):
        undetected_label.append({"a_1": annotation_pair["a_1"], "a_2": annotation_pair["a_2"], "a_3": None})
        if annotation_pair["a_1"] in annotations_1:
            annotations_1.remove(annotation_pair["a_1"])
        if annotation_pair["a_2"] in annotations_2:
            annotations_2.remove(annotation_pair["a_2"])

    # Repeat for annotations_1 and annotations_3
    same, different, undetected = compare_annotations_2(ann_to_polygon_1, ann_to_polygon_3,
                                                        iou_threshold=iou_threshold,
                                                        image_name=image_name + ": a_1, a_3", verbose=verbose)
    for annotation_pair in (same + different):
        undetected_label.append({"a_1": annotation_pair["a_1"], "a_2": None, "a_3": annotation_pair["a_2"]})
        if annotation_pair["a_1"] in annotations_1:
            annotations_1.remove(annotation_pair["a_1"])
        if annotation_pair["a_2"] in annotations_3:
            annotations_3.remove(annotation_pair["a_2"])

    # Repeat for annotations_2 and annotations_3
    same, different, undetected = compare_annotations_2(ann_to_polygon_2, ann_to_polygon_3,
                                                        iou_threshold=iou_threshold,
                                                        image_name=image_name + ": a_2, a_3", verbose=verbose)
    for annotation_pair in (same + different):
        undetected_label.append({"a_1": None, "a_2": annotation_pair["a_1"], "a_3": annotation_pair["a_2"]})
        if annotation_pair["a_1"] in annotations_2:
            annotations_2.remove(annotation_pair["a_1"])
        if annotation_pair["a_2"] in annotations_3:
            annotations_3.remove(annotation_pair["a_2"])

    # Append the remaining annotations to undetected_label accordingly
    for a in annotations_1:
        undetected_label.append({"a_1": a, "a_2": None, "a_3": None})
    for a in annotations_2:
        undetected_label.append({"a_1": None, "a_2": a, "a_3": None})
    for a in annotations_3:
        undetected_label.append({"a_1": None, "a_2": None, "a_3": a})

    if verbose:
        print("---------------")
        print("Same labels: " + str(len(same_label)))
        print("Different labels: " + str(len(different_label)))
        print("Undetected: " + str(len(undetected_label)))
        print("-----------------")
    return same_label, different_label, undetected_label


def create_agreement_matrix(compared_annotation_lists, num_annotators, count_undetected=True, cls_agnostic=False):
    """
    Creates an agreement matrix used for percent agreement.
    :param      compared_annotation_lists: (list dict) List of dictionaries with each entry corresponding to each unique object annotated in each image. The key of each dictionary is the annotator number and the key is the COCO annotation given by that annotator to that object. These can be derived from the compare_annotations_2 and _3 functions.
    :param      num_annotators: (int) Number of annotators per object.
    :param      count_undetected: (bool) If True, include the undetected objects and assign them the category of 0 (the rest of the categories must start from 1).
    :param      cls_agnostic: (bool) If True, count all objects as the same class (useful when isolating segmentation agreement from classification agreement).
    :return:    agreement_matrix: (list) 2D array with the inner list containing the category given to an object by each annotator.
    """
    # Combine all annotations in the annotation lists derived from the compare_annotations functions
    all_objects = []
    for annotations in compared_annotation_lists:
        all_objects.extend(annotations)

    agreement_matrix = []
    for o in all_objects:
        object_categories = []
        # If count_detected is enabled, append 1 for each object if cls_agnostic is enabled, else append the category of each annotation. If an annotator has not annotated an object, append 0 as its category.
        if count_undetected:
            for i in range(num_annotators):
                if o["a_" + str(i + 1)]:
                    if cls_agnostic:
                        object_categories.append(1)
                    else:
                        object_categories.append(o["a_" + str(i + 1)]["category_id"])
                else:
                    object_categories.append(0)
        # If count_detected is enabled, only append categories if the object has been annotated by each annotator.
        else:
            if None not in o.values():
                for i in range(num_annotators):
                    if cls_agnostic:
                        object_categories.append(1)
                    else:
                        object_categories.append(o["a_" + str(i + 1)]["category_id"])

        agreement_matrix.append(object_categories)
    return agreement_matrix


def compute_percent_agreement(agreement_matrix, num_annotators):
    """
    Compute the unweighted percent agreement across annotators.
    :param      agreement_matrix: (list dict) List of dictionaries with each entry corresponding to each unique object annotated in each image. The key of each dictionary is the annotator number and the key is the COCO annotation given by that annotator to that object. These can be derived from the compare_annotations_2 and _3 functions.
    :param      num_annotators: (int) Number of annotators per object.
    :return:    percent_agreement: (float) The percent agreement between annotators.
    """
    all_object_categories = []
    for annotation_set in agreement_matrix:
        # Count the maximum number of matching categories and divide by the number of annotators
        if annotation_set:
            all_object_categories.append(
                annotation_set.count(max(annotation_set, key=annotation_set.count)) / num_annotators)

    percent_agreement = np.mean(all_object_categories)
    return percent_agreement


def compute_krippendorff(annotations, num_annotators, categories, count_undetected=True):
    """
    Computes Krippendorff's Alpha for a set of annotations.
    :param      annotations: (list) 2D array with the inner lists being the annotations from different COCO datasets.
    :param      num_annotators: (int) Number of annotators per object.
    :param      categories: (list) Categories that an object can be assigned.
    :param      count_undetected: (bool) If True, include the undetected objects and assign them the category of 0 (the rest of the categories must start from 1).
    :return:    annotators: (list) 2D array with each array being the categories given by each annotator.
    :return:    k_alpha: (float) Krippendorff's Alpha metric.
    """
    all_objects = []
    for a in annotations:
        all_objects.extend(a)
    categories_given = [[] for a in range(num_annotators)]
    for obj in all_objects:
        if count_undetected:
            for i in range(3):
                if obj["a_" + str(i + 1)] is not None:
                    categories_given[i].append(obj["a_" + str(i + 1)]["category_id"])
                else:
                    categories_given[i].append(0)
        else:
            if None not in obj.values():
                for i in range(3):
                    categories_given[i].append(obj["a_" + str(i + 1)]["category_id"])
    if all(category == categories_given[0] for category in categories_given):
        k_alpha = 1
    else:
        if count_undetected:
            k_alpha = krippendorff.alpha(categories_given, value_domain=categories.insert(0, 0),
                                         level_of_measurement="nominal")
        else:
            k_alpha = krippendorff.alpha(categories_given, value_domain=categories, level_of_measurement="nominal")
    return categories_given, k_alpha


def compute_fleiss(annotations, num_annotators, count_undetected=True, method='uniform'):
    """
    Computes Fleiss' Kappa for a set of annotations.
    :param      annotations: (list) 2D array with the inner lists being the annotations from different COCO datasets.
    :param      num_annotators: (int) Number of annotators per object.
    :param      count_undetected: (bool) If True, include the undetected objects and assign them the category of 0 (the rest of the categories must start from 1).
    :param      method: (string) Method used by the irr.fleiss_kappa function.
    :return:    annotators: (list) 2D array with each array being the categories given by each annotator.
    :return:    f_kappa: (float) Fliess' Kappa metric.
    """
    all_objects = []
    for a in annotations:
        all_objects.extend(a)
    categories_given = [[] for a in range(num_annotators)]
    for obj in all_objects:
        if count_undetected:
            for i in range(3):
                if obj["a_" + str(i + 1)] is not None:
                    categories_given[i].append(obj["a_" + str(i + 1)]["category_id"])
                else:
                    categories_given[i].append(0)
        else:
            if None not in obj.values():
                for i in range(3):
                    categories_given[i].append(obj["a_" + str(i + 1)]["category_id"])
    if all(category == categories_given[0] for category in categories_given):
        f_kappa = 1
    else:
        f_kappa = irr.fleiss_kappa(irr.aggregate_raters(np.array(categories_given).transpose())[0], method=method)
    return categories_given, f_kappa


def compute_some_basic_stats():
    # Compute seed stats across the 5 part dataset

    dataset_path = "../datasets/dataset1/coco/post_quality_check"
    file_name = "_postqc.json"

    seed_stats = {"part1": {},
                  "part2": {},
                  "part3": {},
                  "part4": {},
                  "part5": {}}
    total_images = 0
    total_seeds = 0
    total_viable = 0
    total_nonviable = 0
    total_empty = 0

    for part, stat in seed_stats.items():
        path_to_labels = (os.path.join(dataset_path,
                                       (part + file_name)))
        with open(path_to_labels, 'r') as file:
            data = json.load(file)
        seed_stats[part]["total_images"] = len(data["images"])
        label_num = count_labels(data)[0]
        seed_stats[part]["total_seeds"] = len(data["annotations"])
        seed_stats[part]["seeds_per_image"] = len(data["annotations"]) / len(data["images"])
        seed_stats[part]["viable_num"] = label_num[1]
        seed_stats[part]["nonviable_num"] = label_num[2]
        seed_stats[part]["empty_num"] = label_num[3]
        seed_stats[part]["viable_ratio"] = label_num[1] / len(data["annotations"])
        seed_stats[part]["nonviable_ratio"] = label_num[2] / len(data["annotations"])
        seed_stats[part]["empty_ratio"] = label_num[3] / len(data["annotations"])

        total_images += len(data["images"])
        total_seeds += len(data["annotations"])
        total_viable += label_num[1]
        total_nonviable += label_num[2]
        total_empty += label_num[3]

    # Compute seed stats for the combined, full dataset
    seed_stats["full_dataset"] = {}
    seed_stats["full_dataset"]["total_images"] = total_images
    seed_stats["full_dataset"]["total_seeds"] = total_seeds
    seed_stats["full_dataset"]["seeds_per_image"] = total_seeds / total_images
    seed_stats["full_dataset"]["viable_num"] = total_viable
    seed_stats["full_dataset"]["nonviable_num"] = total_nonviable
    seed_stats["full_dataset"]["empty_num"] = total_empty
    seed_stats["full_dataset"]["viable_ratio"] = total_viable / total_seeds
    seed_stats["full_dataset"]["nonviable_ratio"] = total_nonviable / total_seeds
    seed_stats["full_dataset"]["empty_ratio"] = total_empty / total_seeds

    display(pd.DataFrame(seed_stats))
    # pd.DataFrame(seed_stats).to_csv("../dataset_stats/check2_seed_stats.csv")


def main():
    compute_some_basic_stats()

    # ----------------------------------------------------------------------------

    # Extract all image names and annotations needed for agreement analysis

    # Paths to dataset and corresponding checks
    path_to_part = "../datasets/dataset1/coco/pre_quality_check/all_preqc.json"
    path_to_check_1 = "../datasets/dataset1/coco/check_1/all_check1.json"
    path_to_check_2 = "../datasets/dataset1/coco/check_2/all_check2.json"

    with open(path_to_part, 'r') as file:
        part = json.load(file)

    with open(path_to_check_1, 'r') as file:
        check_1 = json.load(file)

    with open(path_to_check_2, 'r') as file:
        check_2 = json.load(file)

    # Get validation image names from datasets
    agreement_analysis_image_names = []
    if check_1["images"] == check_2["images"]:
        print("All validation images have the same name")
        agreement_analysis_image_names = pd.DataFrame(check_1["images"]).loc[:, "file_name"]
    else:
        print("Check 1 =/= Check 2")

    # Faulty analysis images in pre quality checked dataset (missing annotations, rejected images, etc.)
    rejected_imgs = ["226.jpg", "231.jpg", "261.jpg", "271.jpg", "311.jpg", "316.jpg", "406.jpg", "481.jpg"]
    for rejected_img in rejected_imgs:
        agreement_analysis_image_names.pop(
            agreement_analysis_image_names[agreement_analysis_image_names == rejected_img].index[0])

    # Merge original and check datasets
    all_datasets = [part, check_1, check_2]

    # Dictionary with the keys as the image file name and the values as a list of annotations given by each annotator on said image
    agreement_analysis_annotations = {}
    annotation_counts = [0 for dataset in all_datasets]
    for file_name in agreement_analysis_image_names:

        # Annotations given by each annotator on the same image
        annotations_per_image = []
        for idx, dataset in enumerate(all_datasets):
            # Get the image_id for each file_name in the validation image names
            image_id = next(id["id"] for id in dataset["images"] if id["file_name"] == file_name)
            a = extract_annotations(dataset["annotations"], image_id)
            annotations_per_image.append(a)
            annotation_counts[idx] += len(a)
        agreement_analysis_annotations[file_name] = annotations_per_image
    print("Total number of annotations across all images used for agreement analysis: " + str(sum(annotation_counts)))
    print("Annotation in original: " + str(annotation_counts[0]))
    print("Annotation in check 1: " + str(annotation_counts[1]))
    print("Annotation in check 2: " + str(annotation_counts[2]))

    # ----------------------------------------------------------------------------

    # Compute agreement analysis metrics for different IoU (intersection over union) thresholds

    ious = [0.5]
    all_seed_validations = []
    for i in ious:
        iou_thresh = i
        print(iou_thresh)
        agreement = {}
        all_seed_validations = [[], [], []]
        for v in agreement_analysis_image_names:
            print(v)
            ann_to_polygons = annotations_to_polygons([agreement_analysis_annotations[v][0],
                                                       agreement_analysis_annotations[v][1],
                                                       agreement_analysis_annotations[v][2]])
            validation = compare_annotations_3(ann_to_polygons[0],
                                               ann_to_polygons[1],
                                               ann_to_polygons[2],
                                               iou_thresh, v, False)

            for i in range(3):
                all_seed_validations[i] += validation[i]
            stats = {"same_label": len(validation[0]), "different_label": len(validation[1]),
                     "not_labeled": len(validation[2])}
            kd = compute_krippendorff(validation, 3, [1, 2, 3], count_undetected=True)
            stats["krippendorff_with_undetected"] = kd[1]
            kd = compute_krippendorff(validation, 3, [1, 2, 3], count_undetected=False)
            stats["krippendorff_without_undetected"] = kd[1]
            un = compute_fleiss(validation, 3, count_undetected=True)
            stats["uniform_with_undetected"] = un[1]
            un = compute_fleiss(validation, 3, count_undetected=False)
            stats["uniform_without_undetected"] = un[1]
            p = compute_percent_agreement(
                create_agreement_matrix(validation, 3, count_undetected=True, cls_agnostic=True), 3)
            stats["percentage_agreement_with_undetected"] = p
            p = compute_percent_agreement(create_agreement_matrix(validation, 3, count_undetected=False), 3)
            stats["percentage_agreement_without_undetected"] = p
            agreement[v] = stats

        agreement = pd.DataFrame(agreement)
        agreement['img_mean'] = agreement.mean(axis=1)
        agreement['seed_mean'] = [
            len(all_seed_validations[0]),
            len(all_seed_validations[1]),
            len(all_seed_validations[2]),
            compute_krippendorff(all_seed_validations, 3, [1, 2, 3], count_undetected=True)[1],
            compute_krippendorff(all_seed_validations, 3, [1, 2, 3], count_undetected=False)[1],
            compute_fleiss(all_seed_validations, 3, count_undetected=True)[1],
            compute_fleiss(all_seed_validations, 3, count_undetected=False)[1],
            compute_percent_agreement(
                create_agreement_matrix(all_seed_validations, 3, count_undetected=True, cls_agnostic=True), 3),
            compute_percent_agreement(create_agreement_matrix(all_seed_validations, 3, count_undetected=False), 3)
        ]
        print(agreement.loc[:, ["img_mean", "seed_mean"]])
        # agreement.to_csv(os.path.join("../validation_stats", "new_metrics_" + str(iou_thresh) + ".csv"), index=True)

    all_a = create_agreement_matrix(all_seed_validations, 3, count_undetected=True, cls_agnostic=False)

    all_a_converted = []
    for set_of_three in all_a:
        set_of_three_converted = []
        for a in set_of_three:
            switch = {
                0: 'Undetected',
                1: 'Viable',
                2: 'Non-Viable',
                3: 'Empty'
            }
            set_of_three_converted.append(switch.get(a))
        all_a_converted.append(set_of_three_converted)

    pair_agreement = {}
    for set_of_three in all_a_converted:
        all_combinations = list(itertools.combinations(set_of_three, 2))
        for c in all_combinations:
            pair_agreement[tuple(set(c))] = pair_agreement.get(tuple(set(c)), 0) + 1
    pair_agreement["Total"] = sum(pair_agreement.values())
    pair_agreement = pd.DataFrame(pair_agreement.values(), index=list(pair_agreement.keys()),
                                  columns=["Number of annotations"])
    print(pair_agreement.sort_values("Number of annotations", ascending=False))
    # pair_agreement.to_csv(os.path.join("../validation_stats", "pair_agreement_" + str(iou_thresh) + ".csv"), index=True)

    unique = {}
    for set_of_three in all_a_converted:
        unique[tuple(sorted(set_of_three))] = unique.get(tuple(sorted(set_of_three)), 0) + 1
    unique["Total"] = sum(unique.values())
    unique = pd.DataFrame(unique.values(), index=list(unique.keys()), columns=["Number of annotations"])
    print(unique.sort_values("Number of annotations", ascending=False))
    # unique.to_csv(os.path.join("../validation_stats", "triplet_agreement_" + str(iou_thresh) + ".csv"), index=True)


if __name__ == "__main__":
    main()
