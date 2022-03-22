import cv2
import os
import numpy as np
import array as arr
from pathlib import Path
from matplotlib import pyplot as plt
from PIL import Image
from timeit import default_timer as timer


def overlapping_image(size1, size2):
    (x1, y1, w1, h1), image1 = size1
    (x2, y2, w2, h2), image2 = size2
    if x1 + w1 > x2 and x2 + w2 > x1 and y1 + h1 > y2 and y2 + h2 > y1:
        minx = min(x1, x2)
        miny = min(y1, y2)
        mask = np.zeros((max(y1+h1, y2+h2)-miny, max(x1+w1, x2+w2) - minx), dtype=np.uint8)
        mask[y1 - miny:y1 - miny + h1, x1 - minx:x1 - minx + w1] = image1
        mask[y2 - miny:y2 - miny + h2, x2 - minx:x2 - minx + w2] = image2
        return True, mask
    return False, image1


def get_text_to_image1(image):
    thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)[1]
    dilated_image = cv2.dilate(thresh.copy(), np.ones((10, 1), np.uint8), iterations=1)
    # cv2.imshow('thresh1', dilated_image)
    # cv2.waitKey()
    contours, _ = cv2.findContours(dilated_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dir_images = {}
    list_images = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # print(h)
        # print(x,y,w,h)
        if (h, w) != dilated_image.shape[:2] and h >= 2:
            dir_images.setdefault(x, ((x, y, w, h), thresh[y:y+h, x:x+w]))
    list_img = sorted(dir_images.items(), reverse=False)
    i = 0
    while i <= len(list_img)-2:
        check, img = overlapping_image(list_img[i][1], list_img[i + 1][1])
        list_images.append(cv2.copyMakeBorder(cv2.resize(img, (24, 43)), 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
        if check:
            i += 1
        if i == len(list_img)-2:
            list_images.append(cv2.copyMakeBorder(cv2.resize(list_img[len(list_img)-1][1][1], (24, 43)), 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
        i += 1
    return list_images


def read_dataset(path):
    dataset = []
    for p in Path(path).rglob('*.[jp][pn]*'):
        # try:
        im = Image.open(p.as_posix()).convert('L')  # .convert('L')  # cv2.imread(p.as_posix(), 0)
        # convert image to np arr
        im = np.array(im)
        labels = str(p.name).split('.')[0].split()
        thresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)[1]
        list_images = get_text_to_image1(thresh)
        if len(labels) == len(list_images):
            for i in range(len(labels)):
                dataset.append((labels[i], list_images[i]))
        else:
            print("Fail!")
            break
    return dataset

