import os.path
import unittest
from shapely.geometry import Polygon as ShapelyPolygon

from misc.evaluate import evaluate_segmentations
from shapely.geometry import Polygon
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np


def visualize_polygons(model_polygons, gt_polygons, model_classes=None,
                       gt_classes=None, model_scores=None, save_path=None):
    """
    Visualizes predicted and ground truth polygons in a matplotlib figure.

    :param model_polygons: List of predicted polygon objects
    :param gt_polygons: List of ground-truth polygon objects
    :param model_classes: Optional list of predicted class labels
    :param gt_classes: Optional list of ground-truth class labels
    :param model_scores: Optional list of confidence scores for predictions
    :param save_path: Optional path to save the figure (e.g., 'output.png')
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create patch collections
    gt_patches = []
    model_patches = []

    # Class color mapping (extend as needed)
    class_colors = {
        'viable': 'green',
        'non-viable': 'orange',
        'empty': 'purple',
        None: 'blue'  # Default for unclassified
    }

    # Process ground truth polygons
    for i, poly in enumerate(gt_polygons):
        if poly.is_valid and hasattr(poly, 'exterior'):
            points = list(poly.exterior.coords)
            gt_patches.append(MplPolygon(points, closed=True))

            # Add index tag at top-left corner
            min_x = min(p[0] for p in points)
            max_y = max(p[1] for p in points)
            ax.text(min_x, max_y, f'G{i}',
                    color='black', weight='bold', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round'))

            # Add class label if available
            if gt_classes and i < len(gt_classes):
                centroid = np.mean(points, axis=0)
                cls = gt_classes[i]
                ax.text(centroid[0], centroid[1], f'GT: {cls}',
                        color=class_colors.get(cls, 'blue'),
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Process model polygons with index tags
    for i, poly in enumerate(model_polygons):
        if poly.is_valid and hasattr(poly, 'exterior'):
            points = list(poly.exterior.coords)
            color = class_colors.get(model_classes[i] if model_classes and i < len(model_classes) else None, 'red')

            model_patches.append(MplPolygon(points, closed=True, edgecolor=color))

            # Add index tag at top-left corner
            min_x = min(p[0] for p in points)
            max_y = max(p[1] for p in points)
            ax.text(min_x, max_y, f'M{i}',
                    color='black', weight='bold', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round'))

            # Add score/class label if available
            label_parts = []
            if model_classes and i < len(model_classes):
                label_parts.append(f"Cls: {model_classes[i]}")
            if model_scores and i < len(model_scores):
                label_parts.append(f"Scr: {model_scores[i]:.2f}")

            if label_parts:
                centroid = np.mean(points, axis=0)
                ax.text(centroid[0], centroid[1], '\n'.join(label_parts),
                        color=color, weight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Add collections to the plot
    gt_collection = PatchCollection(gt_patches, facecolor='none',
                                    edgecolor='blue', linewidth=2, alpha=0.7)
    model_collection = PatchCollection(model_patches, facecolor='none',
                                       linewidth=2, alpha=0.7, linestyle='--')

    ax.add_collection(gt_collection)
    ax.add_collection(model_collection)

    # Configure plot
    ax.autoscale_view()
    ax.set_title('Segmentation Comparison\nGround Truth (solid blue) vs Model (dashed colored)')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)

    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Ground Truth'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Model (General)'),
    ]

    # Add class-specific legend entries
    for cls, color in class_colors.items():
        if cls and cls != 'viable':  # Skip default and None
            legend_elements.append(
                Line2D([0], [0], color=color, lw=2, linestyle='--',
                       label=f'Model: {cls}')
            )

    ax.legend(handles=legend_elements, loc='upper right')

    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close(fig)


class TestEvaluateSegmentations(unittest.TestCase):

    def test_class_agnostic_basic_match(self):
        # Setup: Two GT and two Model polygons with IoU > threshold
        gt_polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # Unit square
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])  # Unit square at x=2
        ]
        model_polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # Matches first GT (IoU=1)
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])  # Matches second GT (IoU=1)
        ]
        model_scores = [0.9, 0.8]
        model_classes = ['cls1', 'cls2']
        gt_classes = ['cls1', 'cls2']

        metrics = evaluate_segmentations(
            model_polys, gt_polys, model_classes, gt_classes, model_scores,
            iou_threshold=0.5, confidence_threshold=0.05, cls_agnostic=True
        )

        self.assertEqual(metrics['TP'], 2)
        self.assertEqual(metrics['FP'], 0)
        self.assertEqual(metrics['FN'], 0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)

    def test_class_agnostic_low_confidence(self):
        # Setup: One model has low confidence
        gt_polys = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        model_polys = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])]
        model_scores = [0.9, 0.03]  # Second model below confidence threshold
        model_classes = ['cls1', 'cls2']
        gt_classes = ['cls1']

        metrics = evaluate_segmentations(
            model_polys, gt_polys, model_classes, gt_classes, model_scores,
            iou_threshold=0.5, confidence_threshold=0.05, cls_agnostic=True
        )

        self.assertEqual(metrics['TP'], 1)
        self.assertEqual(metrics['FP'], 0)  # Only one model above threshold (0.9)
        self.assertEqual(metrics['FN'], 0)  # One GT matched

    def test_class_agnostic_duplicate_match(self):
        # Setup: Two GTs match one model (higher IoU wins)
        gt_polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # GT1
            Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)])  # GT2
        ]
        model_polys = [
            Polygon([(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)])  # Model overlaps both
        ]

        model_scores = [0.9]
        model_classes = ['cls1']
        gt_classes = ['cls1', 'cls1']

        visualize_polygons(model_polys, gt_polys, model_classes, gt_classes, model_scores,
                           save_path=os.path.join('outputs', 'test_class_agnostic_duplicate_match.jpg'))

        # Calculate expected IoUs
        inter1 = ShapelyPolygon([(0, 0), (1, 0), (1, 1), (0, 1)]).intersection(
            ShapelyPolygon([(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)])
        ).area
        union1 = ShapelyPolygon([(0, 0), (1, 0), (1, 1), (0, 1)]).union(
            ShapelyPolygon([(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)])
        ).area
        iou1 = inter1 / union1

        inter2 = ShapelyPolygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]).intersection(
            ShapelyPolygon([(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)])
        ).area
        union2 = ShapelyPolygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]).union(
            ShapelyPolygon([(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)])
        ).area
        iou2 = inter2 / union2

        # Ensure higher IoU is chosen
        self.assertGreater(iou1, iou2)

        metrics = evaluate_segmentations(
            model_polys, gt_polys, model_classes, gt_classes, model_scores,
            iou_threshold=0.05, confidence_threshold=0.05, cls_agnostic=True
        )

        self.assertEqual(metrics['TP'], 1)  # Only one match (GT2)
        self.assertEqual(metrics['FN'], 1)  # GT1 unmatched
        self.assertEqual(metrics['FP'], 0)  # Model matched
        self.assertEqual(metrics['TN'], 0)  # Model matched

        metrics = evaluate_segmentations(
            model_polys, gt_polys, model_classes, gt_classes, model_scores,
            iou_threshold=0.5, confidence_threshold=0.05, cls_agnostic=True
        )

        self.assertEqual(metrics['TP'], 0)
        self.assertEqual(metrics['FN'], 2)
        self.assertEqual(metrics['FP'], 1)
        self.assertEqual(metrics['TN'], 0)

    def test_class_specific_basic(self):
        # Setup: Class-specific matching
        gt_polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # viable
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])  # non-viable
        ]
        model_polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # viable
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])  # non-viable
        ]
        model_scores = [0.9, 0.8]
        model_classes = ['viable', 'non-viable']
        gt_classes = ['viable', 'non-viable']

        metrics = evaluate_segmentations(
            model_polys, gt_polys, model_classes, gt_classes, model_scores,
            iou_threshold=0.5, confidence_threshold=0.05, cls_agnostic=False
        )

        # Check viable class
        self.assertEqual(metrics['viable']['TP'], 1)
        self.assertEqual(metrics['viable']['FP'], 0)
        self.assertEqual(metrics['viable']['FN'], 0)
        self.assertEqual(metrics['viable']['precision'], 1.0)
        self.assertEqual(metrics['viable']['recall'], 1.0)

        # Check non-viable class
        self.assertEqual(metrics['non-viable']['TP'], 1)
        self.assertEqual(metrics['non-viable']['FP'], 0)
        self.assertEqual(metrics['non-viable']['FN'], 0)
        self.assertEqual(metrics['non-viable']['precision'], 1.0)
        self.assertEqual(metrics['non-viable']['recall'], 1.0)

        # Check empty class
        self.assertEqual(metrics['empty']['TP'], 0)
        self.assertEqual(metrics['empty']['FP'], 0)
        self.assertEqual(metrics['empty']['FN'], 0)
        self.assertEqual(metrics['empty']['precision'], -1)  # No predictions

    def test_class_specific_mismatch(self):
        # Setup: Class mismatch between GT and model
        gt_polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # viable
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])  # non-viable
        ]
        model_polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # non-viable (wrong)
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])  # viable (wrong)
        ]
        model_scores = [0.9, 0.8]
        model_classes = ['non-viable', 'viable']
        gt_classes = ['viable', 'non-viable']

        metrics = evaluate_segmentations(
            model_polys, gt_polys, model_classes, gt_classes, model_scores,
            iou_threshold=0.5, confidence_threshold=0.05, cls_agnostic=False
        )

        # Viable: GT0 unmatched, Model1 is FP
        self.assertEqual(metrics['viable']['TP'], 0)
        self.assertEqual(metrics['viable']['FP'], 1)  # Model1 predicted 'viable'
        self.assertEqual(metrics['viable']['FN'], 1)  # GT0 unmatched

        # Non-viable: GT1 unmatched, Model0 is FP
        self.assertEqual(metrics['non-viable']['TP'], 0)
        self.assertEqual(metrics['non-viable']['FP'], 1)  # Model0 predicted 'non-viable'
        self.assertEqual(metrics['non-viable']['FN'], 1)  # GT1 unmatched

    def test_class_specific_duplicate_match(self):
        # Setup: Two GTs (same class) match one model
        gt_polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # viable
            Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)])  # viable
        ]
        model_polys = [
            Polygon([(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]),  # viable
            Polygon([(1.25, 1.25), (1.75, 1.25), (1.75, 1.75), (1.25, 1.75)])  # viable (unmatched)
        ]
        model_scores = [0.9, 0.8]
        model_classes = ['viable', 'viable']
        gt_classes = ['viable', 'viable']

        visualize_polygons(model_polys, gt_polys, model_classes, gt_classes, model_scores,
                           save_path=os.path.join('outputs', 'test_class_specific_duplicate_match.jpg'))

        metrics = evaluate_segmentations(
            model_polys, gt_polys, model_classes, gt_classes, model_scores,
            iou_threshold=0.05, confidence_threshold=0.05, cls_agnostic=False
        )

        # One TP (higher IoU match), one FN, one FP (unmatched model)
        self.assertEqual(metrics['viable']['TP'], 1)
        self.assertEqual(metrics['viable']['FP'], 1)  # One unmatched model
        self.assertEqual(metrics['viable']['FN'], 1)  # One unmatched GT
        self.assertEqual(metrics['viable']['precision'], 0.5)  # One unmatched GT


        metrics = evaluate_segmentations(
            model_polys, gt_polys, model_classes, gt_classes, model_scores,
            iou_threshold=0.001, confidence_threshold=0.05, cls_agnostic=False
        )

        # One TP (higher IoU match), one FN, one FP (unmatched model)
        self.assertEqual(metrics['viable']['TP'], 2)
        self.assertEqual(metrics['viable']['FP'], 0)  # One unmatched model
        self.assertEqual(metrics['viable']['FN'], 0)  # One unmatched GT
        self.assertEqual(metrics['viable']['precision'], 1)  # One unmatched GT

    def test_invalid_polygons(self):
        # Setup: Invalid GT polygon
        gt_polys = [Polygon([], is_valid=False)]  # Invalid
        model_polys = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        model_scores = [0.9]
        model_classes = ['cls1']
        gt_classes = ['cls1']

        metrics = evaluate_segmentations(
            model_polys, gt_polys, model_classes, gt_classes, model_scores,
            iou_threshold=0.5, confidence_threshold=0.05, cls_agnostic=True
        )

        self.assertEqual(metrics['TP'], 0)  # No match due to invalid GT
        self.assertEqual(metrics['FP'], 1)  # Model above threshold
        self.assertEqual(metrics['FN'], 1)  # GT not matched


if __name__ == '__main__':
    unittest.main()
