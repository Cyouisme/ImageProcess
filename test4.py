import cv2
import os
import numpy as np
import array as arr
from pathlib import Path
from PIL import ImageChops, Image


def overlapping_image(size1, size2):
    (x1, y1, w1, h1), image1 = size1
    (x2, y2, w2, h2), image2 = size2
    if x1 + w1 > x2 and x2 + w2 > x1 and y1 + w1 > y2 and y2 + w2 > y1:
        minx = min(x1, x2)
        miny = min(y1, y2)
        mask = 255 - np.zeros((max(y1+h1, y2+h2)-miny, max(x1+w1, x2+w2) - minx), dtype=np.uint8)
        mask[y1 - miny:y1 - miny + h1, x1 - minx:x1 - minx + w1] = image1
        mask[y2 - miny:y2 - miny + h2, x2 - minx:x2 - minx + w2] = image2
        return True, mask
    return False, image1


imgDict = {}
for p in Path('images/data').rglob('*.[jp][pn]*'):
    # try:
    image = Image.open(p.as_posix()).convert('L')  # cv2.imread(p.as_posix(), 0)
    # convert image to np arr
    image = np.array(image)
    labels = p.name.split('.')[0].split(' ')
    for label in labels:
        if label == "":
            labels.remove(label)
    thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)[1]
    dilated_image = cv2.dilate(thresh.copy(), np.ones((5, 1), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dir_images = {}
    # list_images = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (h, w) != dilated_image.shape[:2] and h >= 2:
            dir_images.setdefault(x, ((x, y, w, h), 255-thresh[y:y+h, x:x+w]))
    list_img = sorted(dir_images.items(), reverse=False)
    i = 0
    while i < len(list_img)-1:
        check, img = overlapping_image(list_img[i][1], list_img[i + 1][1])
        # list_images.append(img)
        imgDict.update({labels[i]: img})
        if check:
            i += 1
        i += 1

im = cv2.imread('images/imageInput/0.png')
thresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)[1]
dilated = cv2.dilate(thresh.copy(), np.ones((5, 1), np.uint8), iterations=1)
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
dir_images = {}
# list_images = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if (h, w) != dilated_image.shape[:2] and h >= 2:
        dir_images.setdefault(x, ((x, y, w, h), 255-thresh[y:y+h, x:x+w]))
list_img = sorted(dir_images.items(), reverse=False)
i = 0
while i < len(list_img)-1:
    check, im = overlapping_image(list_img[i][1], list_img[i + 1][1])
    # list_images.append(img)
    min_diff = []
    for key, value in imgDict.items():
        res = cv2.absdiff(im, value)
        diff_count = cv2.countNonZero(res)
        if len(min_diff) == 0:
            min_diff = [key, diff_count]
        else:
            if min_diff[1] > diff_count:
                min_diff = [key, diff_count]
    print(min_diff[0], end=' ')
    if check:
        i += 1
    i += 1

# for imgs in list_images:
#     cv2.namedWindow('vv', cv2.WINDOW_NORMAL)
#     cv2.imshow('vv', imgs)
    cv2.waitKey()