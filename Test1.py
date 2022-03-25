import random

import cv2
import numpy as np
from pathlib import Path
from PIL import Image

imgDict = {}


def overlapping_image(texts):
    list_x = []
    list_y = []
    list_xw = []
    list_yh = []
    list_h = []
    for (x, y, w, h), img in texts:
        list_x.append(x)
        list_y.append(y)
        list_xw.append(x + w)
        list_yh.append(y + h)
        list_h.append(h)
    minx = min(list_x)
    miny = min(list_y)
    mask = np.zeros((max(list_yh) - miny, max(list_xw) - minx), dtype=np.uint8)
    for (x, y, w, h), img in texts:
        mask[y - miny:y - miny + h, x - minx:x - minx + w] = img
    return mask


def input_detect():
    for p in Path('images/data').rglob('*.[jp][pn]*'):
        im = Image.open(p.as_posix()).convert('L')
        im = np.array(im)
        labels = p.name.split('.')[0].split(' ')
        for label in labels:
            if label == "":
                labels.remove(label)
        _, thresh = cv2.threshold(im.copy(), 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 30 or h > 15:
                boxes.append([x, 0, x + w, im.shape[0]])
                # cv2.rectangle(im, (x, 0), (x+w, im.shape[0]), (0,255,0), 1)
        boxes = sorted(boxes, key=lambda x: x[0])[:-1]
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            crop_i = thresh[y1:y2, x1:x2]
            idx = np.where(np.all(cv2.cvtColor(crop_i, cv2.COLOR_BGR2GRAY) == (255, 255, 255), axis=2))
            if not len(idx[0]) == 0 and not len(idx[1]) == 0:
                x1, y1, x2, y2 = idx[1].min(), idx[0].min(), idx[1].max(), idx[0].max()
                crop_i = crop_i[y1:y2, x1:x2]
            # Image.fromarray(crop_i).save(f'images/imageOutput/{labels[i]}.png')
                crop_i = cv2.copyMakeBorder(crop_i.copy(), 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                crop_i = cv2.resize(crop_i.copy(), (28, 100))
            # text = []
            # contours, _ = cv2.findContours(crop_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # for c in contours:
            #     x, y, w, h = cv2.boundingRect(c)
            #     if (h, w) != crop_i.shape[:2] and h >= 2:
            #         text.append(((x, y, w, h), crop_i[y:y + h, x:x + w]))
            # crop_i = 255 - cv2.threshold(cv2.resize(overlapping_image(text), (100, 100)), 127, 255, cv2.THRESH_BINARY_INV)[1]
            # crop_i = cv2.copyMakeBorder(crop_i.copy(), 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # Image.fromarray(crop_i).save(f'images/datasetTest/{labels[i]}.png')
            imgDict.update({labels[i]: crop_i})


input_detect()


def output_detect():
    # for f in Path('images/imageInput').rglob('*.[jp][pn]*'):
    #     img = Image.open(f.as_posix()).convert('L')
    #     img = np.array(img)
    img = cv2.imread('images/imageInput/8.png')
    # # img = cv2.resize(img, (900, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img.copy(), 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 30 or h > 15:
            boxes.append([x, 0, x + w, img.shape[0]])
    #         cv2.rectangle(img, (x, 0), (x+w, img.shape[0]), (0,255,0), 1)
    boxes = sorted(boxes, key=lambda x: x[0])
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        crop = thresh[y1:y2, x1:x2]
        idx = np.where(np.all(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) == (255,255,255), axis=2))
        if not len(idx[0]) == 0 and not len(idx[1]) == 0:
            x1, y1, x2, y2 = idx[1].min(), idx[0].min(), idx[1].max(), idx[0].max()
            crop = crop[y1:y2, x1:x2]
            crop = cv2.copyMakeBorder(crop.copy(), 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            crop = cv2.resize(crop.copy(), (28, 100))
        # text = []
        # contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for c in contours:
        #     x, y, w, h = cv2.boundingRect(c)
        #     if (h, w) != crop.shape[:2] and h >= 2:
        #         text.append(((x, y, w, h), crop[y:y + h, x:x + w]))
        # crop = 255 - cv2.threshold(cv2.resize(overlapping_image(text), (100, 100)), 127, 255, cv2.THRESH_BINARY_INV)[1]
        # crop = cv2.copyMakeBorder(crop.copy(), 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # Image.fromarray(crop).save(f'images/imageOutput/{random.random()}.png')
        min_diff = []
        for key, value in imgDict.items():
            # cv2.namedWindow('cr', cv2.WINDOW_NORMAL)
            # cv2.imshow('cr', crop)
            # cv2.namedWindow('value', cv2.WINDOW_NORMAL)
            # cv2.imshow('value', value)
            # cv2.waitKey()
            res = cv2.absdiff(crop, value)
            # cv2.namedWindow('res', cv2.WINDOW_NORMAL)
            # cv2.imshow('res', res)
            # cv2.waitKey()
            diff_count = cv2.countNonZero(res)
            if len(min_diff) == 0:
                min_diff = [key, diff_count]
            else:
                if min_diff[1] > diff_count:
                    min_diff = [key, diff_count]

        print(min_diff[0], end=' ')
    print('\n')


output_detect()
cv2.waitKey()
