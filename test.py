import random

import cv2
import os
import numpy as np
import array as arr
from pathlib import Path
from PIL import ImageChops, Image


def overlapping_image(size1, size2):
    (x1, y1, w1, h1), image1 = size1
    (x2, y2, w2, h2), image2 = size2
    if x1 + w1 > x2 and x2 + w2 > x1 and y1 + h1 > y2 and y2 + h2 > y1:
        minx = min(x1, x2)
        miny = min(y1, y2)
        mask = 255 - np.zeros((max(y1+h1, y2+h2)-miny, max(x1+w1, x2+w2) - minx), dtype=np.uint8)
        mask[y1 - miny:y1 - miny + h1, x1 - minx:x1 - minx + w1] = image1
        mask[y2 - miny:y2 - miny + h2, x2 - minx:x2 - minx + w2] = image2
        return True, mask
    return False, image1


def get_text_to_image(image):
    thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)[1]
    dilated_image = cv2.dilate(thresh.copy(), np.ones((10, 1), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dir_images = {}
    list_images = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (h, w) != dilated_image.shape[:2] and h >= 2:
            dir_images.setdefault(x, ((x, y, w, h), 255-thresh[y:y+h, x:x+w]))
    list_img = sorted(dir_images.items(), reverse=False)
    i = 0
    while i < len(list_img)-1:
        check, img = overlapping_image(list_img[i][1], list_img[i + 1][1])
        list_images.append(img)
        if check:
            i += 1
        i += 1

    for imgs in list_images:
        print(detect_text(imgs))
        # cv2.namedWindow('vv', cv2.WINDOW_NORMAL)
        # cv2.imshow('vv', imgs)
        # cv2.waitKey()
    return list_images


def detect_text(image):
    pass
# for p in Path('images/data').rglob('*.[jp][pn]*'):
#     # try:
#     im = Image.open(p.as_posix())  # .convert('L')  # cv2.imread(p.as_posix(), 0)
#     # convert image to np arr
#     im = np.array(im)
#     labels = p.name.split('.')[0].split(' ')
#     for label in labels:
#         if label == "":
#             labels.remove(label)
#     get_text_to_image(im)
# path = "images/data/"
# arrname = arr.array('i', [])
# arrname = os.listdir(path)
# for i in range(len(arrname)):
#     image = cv2.imread(path+arrname[i], 0)
#     get_text_to_image(image=image)

