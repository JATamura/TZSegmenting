from pickletools import uint8

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import scipy
from skimage.filters import unsharp_mask
import shutil

def process_image(img_path, show=True):
    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if show:
        fig, ax = plt.subplots(2,2)
        ax[0,0 ].imshow(img[:,:,:])
    img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    img = cv2.fastNlMeansDenoisingColored(img, None, 15, 15, 7, 21)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(125,125))
    # img[:,:,0] = clahe.apply(img[:,:,0])
    # img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
    if show:
        ax[0, 1].imshow(img)

    # img = cv2.medianBlur(img, 5)

    # # Sharpen the image
    # img = cv2.filter2D(img, -1, kernel)
    if show:
        ax[1, 0].imshow(img)

    img = np.array(unsharp_mask(img, radius=3, amount=4) * 255, dtype=np.uint8)
    if show:
        ax[1, 1].imshow(img)
        plt.show()

    return img

if __name__ == "__main__":
    # process_image("dataset2/DSC_3988 - Copy.jpg")
    # process_image("dataset1/part1_preqa_coco/images/default/014.jpg")
    process_image("datasets/dataset1/441.jpg")

    # img_path = "dataset1/"
    # for dataset1 in ["p_train/", "p_val/"]:
    #     for file in os.listdir(img_path + dataset1)[:-1]:
    #         os.remove(img_path + dataset1 + file)
    #
    # train_img_path = "dataset1/train/"
    # val_img_path = "dataset1/val/"
    #
    # for image in os.listdir(train_img_path)[:-1]:
    #     processed_image = process_image(train_img_path + image, show = False)
    #     processed_image = cv2.cvtColor(np.array(processed_image*255, dtype=np.float32), cv2.COLOR_RGB2BGR)
    #     cv2.imwrite("dataset1/p_train/" + image, processed_image)
    #
    # for image in os.listdir(val_img_path)[:-1]:
    #     processed_image = process_image(val_img_path + image, show = False)
    #     processed_image = cv2.cvtColor(np.array(processed_image*255, dtype=np.float32), cv2.COLOR_RGB2BGR)
    #     cv2.imwrite("dataset1/p_val/" + image, processed_image)