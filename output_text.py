import cv2
import os
import numpy as np
import array as arr
from pathlib import Path
from PIL import ImageChops, Image
from test5 import read_dataset

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


imgDict = {}
dataset = read_dataset("images/data")
# dataset_upper = read_dataset("images/data")

im = cv2.imread('images/imageInput/8.png')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)[1]
dilated = cv2.dilate(thresh.copy(), np.ones((5, 1), np.uint8), iterations=1)
# cv2.imshow('thresh', dilated)
# cv2.waitKey()
contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
dir_images_lower = {}
dir_images_upper = {}
list_images_lower = []
list_images_upper = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if h >= 42:
        if (h, w) != dilated.shape[:2]:
            dir_images_upper.setdefault(x, ((x, y, w, h), thresh[y:y+h, x:x+w]))
    else:
        if (h, w) != dilated.shape[:2]:
            dir_images_lower.setdefault(x, ((x, y, w, h), thresh[y:y + h, x:x + w]))
list_img_lower = sorted(dir_images_lower.items(), reverse=False)
list_img_upper = sorted(dir_images_upper.items(), reverse=False)
i = 0
while i <= len(list_img_lower)-2:
    check, img = overlapping_image(list_img_lower[i][1], list_img_lower[i + 1][1])
    list_images_lower.append(cv2.copyMakeBorder(cv2.resize(img, (24, 43)), 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    if check:
        i += 1
    if i == len(list_img_lower)-2 and check == False:
        list_images_lower.append(cv2.copyMakeBorder(cv2.resize(list_img_lower[len(list_img_lower)-1][1][1], (24, 43)), 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    i += 1
while i <= len(list_img_upper)-2:
    check, img = overlapping_image(list_img_upper[i][1], list_img_upper[i + 1][1])
    list_images_upper.append(cv2.copyMakeBorder(cv2.resize(img, (24, 43)), 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    if check:
        i += 1
    if i == len(list_img_upper)-2 and check == False:
        list_images_upper.append(cv2.copyMakeBorder(cv2.resize(list_img_upper[len(list_img_upper)-1][1][1], (24, 43)), 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    i += 1
for image in list_images_upper:
    # image = cv2.resize(image, (24, 43))
    min_diff = []
    for image_data in dataset:
        # print(image.shape, image_data[1].shape)
        # cv2.imshow('1', image)
        # cv2.imshow('2', image_data[1])
        res = cv2.absdiff(image, image_data[1])
        diff_count = cv2.countNonZero(res)
        if len(min_diff) == 0:
            min_diff = [image_data[0], diff_count]
        else:
            if min_diff[1] > diff_count:
                min_diff = [image_data[0], diff_count]
    print(min_diff[0], end=' ')
cv2.waitKey()
