import copy
import json
import logging
import os

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
            # if len(annotations["categories"]) > 3:
            #     combined["annotations"][new_annotation_id]["category_id"] -= 1
            if type(combined["annotations"][new_annotation_id]["segmentation"]) != list:
                new_seg = rle_to_coco(combined["annotations"][new_annotation_id])[0]
                combined["annotations"][new_annotation_id]["segmentation"] = [new_seg["segmentation_coords"]]

        image_id += len(annotations["images"])
        annotation_id += len(annotations["annotations"])

    with open(output_path, "w") as outfile:
        json.dump(combined, outfile)

def extract_annotations(all_annotations, image_id):
    image_annotations = []
    for annotation in all_annotations:
        if annotation["image_id"] == image_id:
            image_annotations.append(annotation)
    return image_annotations

if __name__ == "__main__":
    # Merge dataset
    # datasets_to_merge = ["../datasets/dataset1/coco/pre_quality_check/part1_preqc.json"]
    # output_path = "../datasets/dataset1/coco/pre_quality_check/part1_preqc.json"
    # combine_datasets(datasets_to_merge, output_path)
    # datasets_to_merge = ["../datasets/dataset1/coco/pre_quality_check/part2_preqc.json"]
    # output_path = "../datasets/dataset1/coco/pre_quality_check/part2_preqc.json"
    # combine_datasets(datasets_to_merge, output_path)
    # datasets_to_merge = ["../datasets/dataset1/coco/pre_quality_check/part3_preqc.json"]
    # output_path = "../datasets/dataset1/coco/pre_quality_check/part3_preqc.json"
    # combine_datasets(datasets_to_merge, output_path)
    # datasets_to_merge = ["../datasets/dataset1/coco/pre_quality_check/part4_preqc.json"]
    # output_path = "../datasets/dataset1/coco/pre_quality_check/part4_preqc.json"
    # combine_datasets(datasets_to_merge, output_path)
    # datasets_to_merge = ["../datasets/dataset1/coco/pre_quality_check/part5_preqc.json"]
    # output_path = "../datasets/dataset1/coco/pre_quality_check/part5_preqc.json"
    # combine_datasets(datasets_to_merge, output_path)

    datasets_to_merge = ["../datasets/dataset1/coco/pre_quality_check/part1_preqc.json",
                         "../datasets/dataset1/coco/pre_quality_check/part2_preqc.json",
                         "../datasets/dataset1/coco/pre_quality_check/part3_preqc.json",
                         "../datasets/dataset1/coco/pre_quality_check/part4_preqc.json",
                         "../datasets/dataset1/coco/pre_quality_check/part5_preqc.json"]
    output_path = "../datasets/dataset1/coco/pre_quality_check/all_preqc.json"
    print(output_path)
    combine_datasets(datasets_to_merge, output_path)

    datasets_to_merge = ["../datasets/dataset1/coco/post_quality_check/part1_postqc.json",
                         "../datasets/dataset1/coco/post_quality_check/part2_postqc.json",
                         "../datasets/dataset1/coco/post_quality_check/part3_postqc.json",
                         "../datasets/dataset1/coco/post_quality_check/part4_postqc.json",
                         "../datasets/dataset1/coco/post_quality_check/part5_postqc.json"]
    output_path = "../datasets/dataset1/coco/post_quality_check/all_postqc.json"
    print(output_path)
    combine_datasets(datasets_to_merge, output_path)

    # ...and corresponding checks
    datasets_to_merge = ["../datasets/dataset1/coco/check_1/part1_check1.json",
                         "../datasets/dataset1/coco/check_1/part2_check1.json",
                         "../datasets/dataset1/coco/check_1/part3_check1.json",
                         "../datasets/dataset1/coco/check_1/part4_check1.json",
                         "../datasets/dataset1/coco/check_1/part5_check1.json"]
    output_path = "../datasets/dataset1/coco/check_1/all_check1.json"
    print(output_path)
    combine_datasets(datasets_to_merge, output_path)

    datasets_to_merge = ["../datasets/dataset1/coco/check_2/part1_check2.json",
                         "../datasets/dataset1/coco/check_2/part2_check2.json",
                         "../datasets/dataset1/coco/check_2/part3_check2.json",
                         "../datasets/dataset1/coco/check_2/part4_check2.json",
                         "../datasets/dataset1/coco/check_2/part5_check2.json"]
    output_path = "../datasets/dataset1/coco/check_2/all_check2.json"
    print(output_path)
    combine_datasets(datasets_to_merge, output_path)