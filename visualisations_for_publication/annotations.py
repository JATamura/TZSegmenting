import os
import pathlib

import cv2
import numpy as np
import matplotlib.pyplot as plt
import shapely
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon

from agreement_analysis.analysis import get_agreement_analysis_annotations, annotations_to_polygons, compare_annotations_3
from utils import REPO_PATH

colors = {'viable': 'red', 'nonviable': 'green', 'empty': 'black'}


def polygon_iou(a1_polygon, a2_polygon, a3_polygon):
    intersection = shapely.intersection_all(
        [a1_polygon, a2_polygon, a3_polygon]).area
    union = shapely.union_all([a1_polygon, a2_polygon, a3_polygon]).area
    iou = intersection / union
    return iou


def plot_bbox(ax, annotation, color, linestyle='--', linewidth=2):
    """
    Plot a COCO-format bbox from an annotation.
    """
    if annotation is None or "bbox" not in annotation:
        return

    x, y, w, h = annotation["bbox"]
    rect = Rectangle(
        (x, y),
        w,
        h,
        fill=False,
        edgecolor=color,
        linestyle=linestyle,
        linewidth=linewidth
    )
    ax.add_patch(rect)


def plot_class_examples(same_label, img, img_name):
    viables = []
    nonviables = []
    empties = []
    for a_group in same_label:
        if a_group['a_1']['category_id'] == 1:
            viables.append(a_group)
        elif a_group['a_1']['category_id'] == 2:
            nonviables.append(a_group)
        elif a_group['a_1']['category_id'] == 3:
            empties.append(a_group)
        else:
            print('Unknown category id:', a_group['a_1']['category_id'])

    if len(viables) == 0 or len(nonviables) == 0 or len(empties) == 0:
        return
    out_dir = os.path.join('outputs', 'class examples', img_name.strip('.jpg'))
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    for i_viable in viables:
        out_file = os.path.join(out_dir, f'viable_{viables.index(i_viable)}.jpg')
        plot_three_segmentations(i_viable['a_1'], i_viable['a_2'], i_viable['a_3'], img, False, out_file)
    for i_nonviable in nonviables:
        out_file = os.path.join(out_dir, f'nonviable_{nonviables.index(i_nonviable)}.jpg')
        plot_three_segmentations(i_nonviable['a_1'], i_nonviable['a_2'], i_nonviable['a_3'], img, False, out_file)
    for i_empty in empties:
        out_file = os.path.join(out_dir, f'empty_{empties.index(i_empty)}.jpg')
        plot_three_segmentations(i_empty['a_1'], i_empty['a_2'], i_empty['a_3'], img, False, out_file)


def combine_examples():
    raise NotImplementedError

    def resize_to_height(img, height):
        """Resize an image to a given height, preserving aspect ratio."""
        h, w = img.shape[:2]
        new_w = int(round(w * height / h))
        return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)

    def resize_to_width(img, width):
        """Resize an image to a given width, preserving aspect ratio."""
        h, w = img.shape[:2]
        new_h = int(round(h * width / w))
        return cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)

    # Combine examples
    # take image whole image '051.jpg' and show underneath empty_0.jpg, empty_1.jpg, empty_2.jpg
    img_path = '../datasets/dataset1/all_images/051.jpg'
    small_paths = [
        os.path.join('outputs','class examples', '051', 'viable_6.jpg'),
        os.path.join('outputs','class examples', '051', 'nonviable_8.jpg'),
        os.path.join('outputs','class examples', '051', 'empty_1.jpg'),
    ]

    main_image = cv2.imread(img_path)
    small_images = [cv2.imread(p) for p in small_paths]
    # Row 2: make the three small images the same height, with gaps between them
    row_height = min(im.shape[0] for im in small_images)
    small_images = [resize_to_height(im, row_height) for im in small_images]

    gap = 0
    bg_color = (255, 255, 255)

    spacer = np.full((row_height, gap, 3), bg_color, dtype=np.uint8)
    bottom_row = small_images[0]
    for im in small_images[1:]:
        bottom_row = np.concatenate((bottom_row, spacer, im), axis=1)

    # Make both rows the same width (scale the bottom row to match the main image)
    # bottom_row = resize_to_width(bottom_row, main_image.shape[1])
    #
    # # Stack: main image on top, gap, then the row of three
    # v_spacer = np.full((gap, main_image.shape[1], 3), bg_color, dtype=np.uint8)
    # combined_img = np.concatenate((main_image, v_spacer, bottom_row), axis=0)
    cv2.imwrite(os.path.join('outputs','class examples', 'combined_examples.jpg'), bottom_row, )


def plot_three_segmentations(a_1, a_2, a_3, img, use_class_colors, out_file):
    fig, ax = plt.subplots()

    def convert_to_plygon(a):
        a_seg = np.array(a["segmentation"][0])
        polygon = Polygon(np.array(a_seg).reshape(int(len(a_seg) / 2), 2))
        return polygon

    set_colors = ['red', 'green', 'blue']
    global color_i
    color_i = -1

    def get_color(a):
        if use_class_colors:
            switch = {
                1: 'viable',
                2: 'nonviable',
                3: 'empty'
            }

            return colors[switch[a['category_id']]]
        else:
            global color_i
            color_i += 1
            return set_colors[color_i]

    # a_1
    poly1 = convert_to_plygon(a_1)
    ax.plot(*poly1.exterior.xy, c=get_color(a_1))
    # plot_bbox(ax, a_1, color_1)

    # a_2
    poly2 = convert_to_plygon(a_2)
    ax.plot(*poly2.exterior.xy, c=get_color(a_2))
    # plot_bbox(ax, a_2, 'green')

    # a_3
    poly3 = convert_to_plygon(a_3)
    ax.plot(*poly3.exterior.xy, c=get_color(a_3))
    # plot_bbox(ax, a_3, 'blue')

    iou = polygon_iou(poly1, poly2, poly3)

    x1, y1, w1, h1 = a_1["bbox"]
    x2, y2, w2, h2 = a_2["bbox"]
    x3, y3, w3, h3 = a_3["bbox"]

    text_x = min(x1, x2, x3)
    text_y = min(y1, y2, y3)

    min_x = min(x1, x2, x3, poly1.bounds[0], poly2.bounds[0], poly3.bounds[0])
    min_y = min(y1, y2, y3, poly1.bounds[1], poly2.bounds[1], poly3.bounds[1])
    max_x = max(x1 + w1, x2 + w2, x3 + w3, poly1.bounds[2], poly2.bounds[2], poly3.bounds[2])
    max_y = max(y1 + h1, y2 + h2, y3 + h3, poly1.bounds[3], poly2.bounds[3], poly3.bounds[3])

    ax.text(
        text_x,
        text_y,
        f"IoU: {iou:.3f}",
        color="black",
        fontsize=12,
        weight="bold",
        bbox=dict(
            facecolor="white",
            alpha=0.8,
            edgecolor="black",
            boxstyle="round"
        )
    )

    ax.imshow(img)

    pad = 20
    ax.set_xlim([min_x - pad, max_x + pad])
    ax.set_ylim([max_y + pad, min_y - pad])

    plt.axis('off')
    plt.savefig(out_file, bbox_inches='tight', pad_inches=0, dpi=300)

    plt.close()
    plt.cla()
    plt.clf()


def run_annotation_plots():
    from PIL import Image
    image_path = os.path.join(REPO_PATH, "datasets/dataset1/all_images")

    agreement_analysis_image_names, agreement_analysis_annotations, output_path = get_agreement_analysis_annotations('post')
    for image_name in agreement_analysis_image_names.tolist()[:5]:
        iou_thresh = 0.3
        print(image_name)

        # Open the image file

        image = Image.open(os.path.join(image_path, image_name))

        ann_to_polygons = annotations_to_polygons([agreement_analysis_annotations[image_name][0],
                                                   agreement_analysis_annotations[image_name][1],
                                                   agreement_analysis_annotations[image_name][2]])
        same_label, different_label, undetected_label = compare_annotations_3(ann_to_polygons[0],
                                                                              ann_to_polygons[1],
                                                                              ann_to_polygons[2],
                                                                              iou_thresh, image_name, False)
        plot_class_examples(same_label, image, image_name)


def main():
    run_annotation_plots()


if __name__ == '__main__':
    main()
