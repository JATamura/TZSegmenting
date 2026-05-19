import os
import json
import numpy as np
import shapely
from shapely.geometry import Polygon
import krippendorff
import copy
import pandas as pd
from statsmodels.stats import inter_rater as irr
from utils import extract_annotations
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


def create_agreement_matrix(compared_annotation_lists, count_undetected=True, cls_agnostic=False):
    """
    Creates an agreement matrix used for percent agreement.
    :param      compared_annotation_lists: (list dict) List of dictionaries with each entry corresponding to each unique object annotated in each image. The key of each dictionary is the annotator number and the key is the COCO annotation given by that annotator to that object. These can be derived from the compare_annotations_2 and _3 functions.
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
            for a in o:
                if o[a] is not None:
                    if cls_agnostic:
                        object_categories.append(1)
                    else:
                        object_categories.append(o[a]["category_id"])
                else:
                    object_categories.append(0)
        # If count_detected is enabled, only append annotated objects
        else:
            # if None not in o.values():
            for a in o:
                if o[a] is not None:
                    if cls_agnostic:
                        object_categories.append(1)
                    else:
                        object_categories.append(o[a]["category_id"])

        if len(object_categories) > 0:
            agreement_matrix.append(object_categories)
    return agreement_matrix


def compute_percent_agreement(agreement_matrix):
    """
    Compute the unweighted percent agreement across annotators.

    Calculated as in Table 2 of https://pmc.ncbi.nlm.nih.gov/articles/PMC3900052/
    Assumes majority are correct.

    :param      agreement_matrix: (list dict) List of dictionaries with each entry corresponding to each unique object annotated in each image. The key of each dictionary is the annotator number and the key is the COCO annotation given by that annotator to that object. These can be derived from the compare_annotations_2 and _3 functions.
    :return:    percent_agreement: (float) The percent agreement between annotators.
    """

    all_object_categories = []
    for annotation_set in agreement_matrix:
        # Count the maximum number of matching categories and divide by the number of annotators
        if annotation_set and len(annotation_set) > 1:
            majority_category = max(annotation_set, key=annotation_set.count)
            percent_agreement_for_annotation = annotation_set.count(majority_category) / len(annotation_set)
            all_object_categories.append(percent_agreement_for_annotation)

    percent_agreement = np.mean(all_object_categories)
    return percent_agreement


def compute_probability_of_agreement(agreement_matrix):
    '''
    Given a set of annotations, this calculates the probability of agreement if you had randomly selected an annotation and randomly selected two annotators.
    :param agreement_matrix:
    :return:
    '''

    agreement_count = 0
    disagreement_count = 0
    for set_of_three in agreement_matrix:
        all_combinations = list(itertools.combinations(set_of_three, 2))
        for c in all_combinations:
            if c[0] == c[1]:
                agreement_count += 1
            else:
                disagreement_count += 1
    agreement_probability = agreement_count / (agreement_count + disagreement_count)
    return agreement_probability


def compute_krippendorff(annotations, num_annotators, categories):
    """
    Computes Krippendorff's Alpha for a set of annotations.
    :param      annotations: (list) 2D array with the inner lists being the annotations from different COCO datasets.
    :param      num_annotators: (int) Number of annotators per object.
    :param      categories: (list) Categories that an object can be assigned.
    :param      class_agnostic_for_detection: (bool) If True, just assess detection of seeds independent of class.
    :return:    annotators: (list) 2D array with each array being the categories given by each annotator.
    :return:    k_alpha: (float) Krippendorff's Alpha metric.
    """
    all_objects = []
    for a in annotations:
        all_objects.extend(a)
    categories_given = [[] for a in range(num_annotators)]
    for obj in all_objects:
        for i in range(num_annotators):
            if obj["a_" + str(i + 1)] is not None:
                categories_given[i].append(obj["a_" + str(i + 1)]["category_id"])
            else:
                categories_given[i].append(np.nan)
    if all(category == categories_given[0] for category in categories_given):
        k_alpha = 1
    else:
        k_alpha = krippendorff.alpha(categories_given, value_domain=categories, level_of_measurement="nominal")
    return k_alpha


def _compute_randolph(annotations, num_annotators, count_undetected=True, method='fleiss'):
    """
    Computes Randolph's Kappa for a set of annotations. This can be changed to Fleiss' Kappa by changing the 'method' parameter.

    The percentage agreement we use + fleiss' or kripendorff provide better overall agreement metrics.
    :param      annotations: (list) 2D array with the inner lists being the annotations from different COCO datasets.
    :param      num_annotators: (int) Number of annotators per object.
    :param      count_undetected: (bool) If True, include the undetected objects and assign them the category of 0 (the rest of the categories must start from 1).
    :param      method: (string) Method used by the irr.fleiss_kappa function.
    :return:    annotators: (list) 2D array with each array being the categories given by each annotator.
    :return:    f_kappa: (float) Randolph's Kappa metric.
    """
    raise NotImplementedError(
        "Using 'uniform' distribution doesn't make a huge amount of sense as it is accounting for probablity of chance agreement while miscalculating chance agreement.")
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


def get_agreement_analysis_annotations(pre_or_post: str):
    # Extract all image names and annotations needed for agreement analysis

    # # Paths to pre-quality checked datasets and corresponding duplicates
    if pre_or_post == 'pre':
        path_to_part = "../datasets/dataset1/coco_format/pre_quality_check/base_dataset.json"
        path_to_check_1 = "../datasets/dataset1/coco_format/pre_quality_check/analysis_duplicate_1.json"
        path_to_check_2 = "../datasets/dataset1/coco_format/pre_quality_check/analysis_duplicate_2.json"
        output_path = "pre_quality_check"

    if pre_or_post == 'post':
        # # Paths to post-quality checked datasets and corresponding duplicates
        path_to_part = "../datasets/dataset1/coco_format/post_quality_check/base_dataset.json"
        path_to_check_1 = "../datasets/dataset1/coco_format/post_quality_check/analysis_duplicate_1.json"
        path_to_check_2 = "../datasets/dataset1/coco_format/post_quality_check/analysis_duplicate_2.json"
        output_path = "post_quality_check"

    with open(path_to_part, 'r') as file:
        part = json.load(file)

    with open(path_to_check_1, 'r') as file:
        check_1 = json.load(file)

    with open(path_to_check_2, 'r') as file:
        check_2 = json.load(file)

    # Get validation image names from datasets
    # This script assumes all images are in separate json files and the corresponding images have identical image names
    agreement_analysis_image_names = []
    if check_1["images"] == check_2["images"]:
        print("All validation images have the same name")
        agreement_analysis_image_names = pd.DataFrame(check_1["images"]).loc[:, "file_name"]
    else:
        print("Check 1 =/= Check 2")

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
        if [] in annotations_per_image:
            print(file_name + " has no annotations at index " + str(
                [i for i, annotations in enumerate(annotations_per_image) if
                 annotations == []]) + " and was removed from analysis")
        # Append to the dictionary the file name and corresponding number of annotations made by each annotator
        else:
            agreement_analysis_annotations[file_name] = annotations_per_image
            for i, annotations in enumerate(annotations_per_image):
                annotation_counts[i] += len(annotations)

    print("Total number of annotations across all images used for agreement analysis: " + str(sum(annotation_counts)))
    print("Annotation in original: " + str(annotation_counts[0]))
    print("Annotation in check 1: " + str(annotation_counts[1]))
    print("Annotation in check 2: " + str(annotation_counts[2]))

    return agreement_analysis_image_names, agreement_analysis_annotations, output_path


def main(pre_or_post: str):
    agreement_analysis_image_names, agreement_analysis_annotations, output_path = get_agreement_analysis_annotations(pre_or_post)
    # ----------------------------------------------------------------------------

    # Compute agreement analysis metrics for different IoU (intersection over union) thresholds

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # The threshold for determining if an annotation was for the same seed or not was initially set to 0.5 but later changed to 0.3 as there
    # was too much variance in how people annotated (even after quality checking) that even 0.5 was too high of a threshold to consider
    # different annotations made as the same seed.
    ious = [0.3]
    for iou_thresh in ious:
        print(iou_thresh)
        agreement = {}
        all_seed_validations = [[], [], []]
        for image_name in agreement_analysis_image_names:
            if agreement_analysis_annotations.get(image_name) is None:
                print(image_name)
                continue
            print(image_name)
            ann_to_polygons = annotations_to_polygons([agreement_analysis_annotations[image_name][0],
                                                       agreement_analysis_annotations[image_name][1],
                                                       agreement_analysis_annotations[image_name][2]])
            validation = compare_annotations_3(ann_to_polygons[0],
                                               ann_to_polygons[1],
                                               ann_to_polygons[2],
                                               iou_thresh, image_name, False)

            for i in range(3):
                all_seed_validations[i] += validation[i]
            stats = {"same_label": len(validation[0]), "different_label": len(validation[1]),
                     "not_labeled": len(validation[2])}
            stats["classification_krippendorff"] = compute_krippendorff(validation, 3, [1, 2, 3])
            # un = compute_randolph(validation, 3, count_undetected=True)
            # stats["uniform_with_undetected"] = un[1]
            # un = compute_randolph(validation, 3, count_undetected=False)
            # stats["uniform_without_undetected"] = un[1]

            detection_agreement_matrix = create_agreement_matrix(validation, count_undetected=True, cls_agnostic=True)
            classification_agreement_matrix = create_agreement_matrix(validation, count_undetected=False)

            stats["pure_detection_percent_agreement"] = compute_percent_agreement(
                detection_agreement_matrix)  # Get a measure of pure detection agreement agnostic of class
            stats["pure_classification_percent_agreement"] = compute_percent_agreement(
                classification_agreement_matrix)  # And a measure of pure classification agreement agnostic of detection
            stats["pure_detection_probability_of_agreement"] = compute_probability_of_agreement(detection_agreement_matrix)
            stats["pure_classification_probability_of_agreement"] = compute_probability_of_agreement(classification_agreement_matrix)

            agreement[image_name] = stats

        agreement = pd.DataFrame(agreement)
        agreement['per_img_mean'] = agreement.mean(axis=1)

        all_seed_classification_agreement_matrix = create_agreement_matrix(all_seed_validations, count_undetected=False)
        all_seed_detection_agreement_matrix = create_agreement_matrix(all_seed_validations, count_undetected=True, cls_agnostic=True)
        agreement['dataset_total'] = [
            len(all_seed_validations[0]),
            len(all_seed_validations[1]),
            len(all_seed_validations[2]),
            compute_krippendorff(all_seed_validations, 3, [1, 2, 3]),
            # compute_randolph(all_seed_validations, 3, count_undetected=True)[1],
            # compute_randolph(all_seed_validations, 3, count_undetected=False)[1],
            compute_percent_agreement(all_seed_detection_agreement_matrix),
            compute_percent_agreement(all_seed_classification_agreement_matrix),
            compute_probability_of_agreement(all_seed_detection_agreement_matrix),
            compute_probability_of_agreement(all_seed_classification_agreement_matrix)
        ]
        print(agreement.loc[:, ["per_img_mean", "dataset_total"]])
        agreement.to_csv(os.path.join(output_path, "metrics_" + str(iou_thresh) + ".csv"), index=True)

        all_a = create_agreement_matrix(all_seed_validations, count_undetected=True, cls_agnostic=False)

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
                order = tuple(set(c))
                pair_agreement[order] = pair_agreement.get(order, 0) + 1
        pair_agreement["Total"] = sum(pair_agreement.values())
        pair_agreement = pd.DataFrame(pair_agreement.values(), index=list(pair_agreement.keys()),
                                      columns=["Number of annotations"])
        print(pair_agreement.sort_values("Number of annotations", ascending=False))
        pair_agreement.to_csv(os.path.join(output_path, "pair_agreement_" + str(iou_thresh) + ".csv"), index=True)

        unique = {}
        for set_of_three in all_a_converted:
            unique[tuple(sorted(set_of_three))] = unique.get(tuple(sorted(set_of_three)), 0) + 1
        unique["Total"] = sum(unique.values())
        unique = pd.DataFrame(unique.values(), index=list(unique.keys()), columns=["Number of annotations"])
        print(unique.sort_values("Number of annotations", ascending=False))
        unique.to_csv(os.path.join(output_path, "triplet_agreement_" + str(iou_thresh) + ".csv"), index=True)


if __name__ == "__main__":
    main('pre')
    main('post')
