import imutils
import numpy as np
import cv2
from pathlib import Path
from PIL import ImageChops, Image
import random
from imutils import contours


# Compare two images (image not numpy array
def equal(im1, im2):
    return ImageChops.difference(im1, im2).getbbox()


# Convert numpy arr to image
def convert_image(im_convert):
    im_convert = Image.fromarray(im_convert)
    return im_convert


# function to return key for any value
def get_key(val):
    for key, value in imgDict.items():
        if val == value:
            return key

    return "key doesn't exist"

# Read and fix input image
imgDict = {}
# đọc hết tất cả các ảnh từ file
for p in Path('images/data').rglob('*.[jp][pn]*'):
    # try:
    # .convert('L')  # cv2.imread(p.as_posix(), 0)
    im = Image.open(p.as_posix())
    # convert image to np arr
    im = np.array(im)
    labels = p.name.split('.')[0].split(' ')
    for label in labels:
        if label == "":
            labels.remove(label)
    # chuyển ảnh thành nền đen chữ trắng
    _, thresh = cv2.threshold(cv2.cvtColor(
        im.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    # enumerate: thay thế i=0,i++
    for (i, c) in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(c)
        if w > 30 or h > 15:
            boxes.append([x, 0, x + w, im.shape[0]])
    boxes = sorted(boxes, key=lambda x: x[0])
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        crop_i = im[y1:y2, x1:x2]
        Image.fromarray(crop_i).save(f'images/imageF/{labels[i]}.png')
        imgDict.update({labels[i]: crop_i})
        # print(labels[i])
        # cv2.imshow('ff', crop)
        # cv2.waitKey()
    # except:
    #     cv2.imshow('tt', im)
    #     cv2.waitKey()

#Read and compare output image with fixed image
im = cv2.imread('images/imageInput/9.png')
imgray = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2GRAY)
cv2.imshow('imgray', imgray)
kernel = np.ones((2, 2), np.uint8)
dilation = cv2.dilate(imgray.copy(), kernel, iterations=1)
cv2.imshow('dilation', dilation)
# Nhị phân hoá ảnh với THRESH_BINARY_INV để chuyển đối tượng có màu đen nền trắng sang ngược lại
_, thresh = cv2.threshold(dilation.copy(), 127, 255, cv2.THRESH_BINARY_INV)
# Tìm countour của ảnh - contour chỉ được xác định trên đối tượng có màu trắng nền đen
contours, _ = cv2.findContours(
    thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.imshow('cc', thresh)

boxes = []
for (i, c) in enumerate(contours):
    x, y, w, h = cv2.boundingRect(c)
    # cv2.rectangle(im, (x, y), (x + w), (255, 0, 0), 1)
    if w > 5 or h > 2:
        boxes.append([x, 0, x + w, im.shape[0]])
boxes = sorted(boxes, key=lambda x: x[0])
for i, (x1, y1, x2, y2) in enumerate(boxes):
    crop = im[y1:y2, x1:x2]
    Image.fromarray(crop).save(f'images/imageOutput/{random.random()}.png')
    idx = np.where(np.all(crop == (34, 34, 34), axis=2))
    if not len(idx[0]) == 0 and not len(idx[1]) == 0:
        x1, y1, x2, y2 = idx[1].min(), idx[0].min(), idx[1].max(), idx[0].max()
        # cv2.rectangle(value, (x1, y1), (x2, y2), (0, 0, 0), 2)
        crop = crop[y1:y2, x1:x2]
    crop = Image.fromarray(np.array(crop)).resize((16, 50))
    # crop.show()
    for value in imgDict.values():  # đây nè, t lặp một vòng lặp trong cái imgDict để lấy cái value của hắn ra
        idx = np.where(np.all(value == (34, 34, 34), axis=2))
        if not len(idx[0]) == 0 and not len(idx[1]) == 0:
            x1, y1, x2, y2 = idx[1].min(), idx[0].min(
            ), idx[1].max(), idx[0].max()
            # cv2.rectangle(value, (x1, y1), (x2, y2), (0, 0, 0), 2)
            value = value[y1:y2, x1:x2]
        value = Image.fromarray(value)
        value = value.resize((16, 50))
        # value.show()
        if equal(value, crop) is None:
            print(get_key(value))
        else:
            continue
    # for j in range(30):
    #     Image.fromarray(crop).save(f'images/imageOutput/{random.random()}.png')

cv2.waitKey()
