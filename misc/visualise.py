import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import Polygon

def plot_segmentation(a_1, a_2=None, a_3=None, ax=None, img=None):
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    if not a_1 is None:
        a_1_seg = np.array(a_1["segmentation"][0])
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
        a_2_seg = np.array(a_2["segmentation"][0])
        p2 = np.array(a_2_seg).reshape(int(len(a_2_seg)/2), 2)
        poly2 = Polygon(p2)
        ax.plot(*poly2.exterior.xy, c='green')
    if not a_3 is None:
        a_3_seg = np.array(a_3["segmentation"][0])
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
