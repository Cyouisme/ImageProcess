import random

import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from PIL import Image


def overlapping_image(size1, size2):
    (x1, y1, w1, h1), image1 = size1
    (x2, y2, w2, h2), image2 = size2
    if x1 + w1 >= x2 and x2 + w2 >= x1 and y1 + h1 >= y2 and y2 + h2 >= y1:
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
            dir_images.setdefault(x, ((x, y, w, h), image[y:y+h, x:x+w]))
    list_img = sorted(dir_images.items(), reverse=False)
    i = 0
    while i <= len(list_img)-2:
        check, img = overlapping_image(list_img[i][1], list_img[i + 1][1])
        list_images.append(img)
        if check:
            i += 1
        if i == len(list_img)-2 and check == False:
            list_images.append(list_img[len(list_img)-1][1][1])
        i += 1
    return list_images


def create_dataset(image, paths):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    list_images = get_text_to_image(image)
    print(len(paths) == len(list_images))
    if len(paths) == len(list_images):
        for i in range(len(paths)):
            plt.imsave("../dataset/"+paths[i]+".png", list_images[i])


path = "images/imageInput"
for p in Path(path).rglob('*.[jp][pn]*'):
    # try:
    im = Image.open(p.as_posix())  # .convert('L')  # cv2.imread(p.as_posix(), 0)
    # convert image to np arr
    im = np.array(im)
    labels = str(p.name).split('.')[0].split()
    create_dataset(image=im, paths=labels)
