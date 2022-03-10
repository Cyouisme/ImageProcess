import numpy as np
import cv2
from pathlib import Path
from PIL import ImageChops, Image
#Hàm so sánh 2 ảnh
# def equal(im1, im2):
#     return ImageChops.difference(im1, im2).getbbox() is None
imgDict = {}
images = []
#đọc hết tất cả các ảnh từ file
for p in Path('data').rglob('*.[jp][pn]*'):
    try:
        im = Image.open(p.as_posix()).convert('L')#cv2.imread(p.as_posix(), 0)
        im = np.array(im)
        labels = p.name.split('.')[0].split(' ')
        for label in labels:
            if label == "":
                labels.remove(label)
        # chuyển ảnh thành nền đen chữ trắng
        _, thresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append([x, 0, x + w, im.shape[0]])
        boxes = sorted(boxes, key=lambda x: x[0])
        crop = im[boxes[0][1]:boxes[0][3], boxes[0][0]:boxes[0][2]]
        # cv2.imshow('ff', crop)
        for i in range(15):
            imgDict.update({"" + labels[i]: crop})
        # cv2.waitKey()
        images.append(im)
    except:
        cv2.imshow('tt',im)
        cv2.waitKey()
# cv2.imshow('zz', images[0])
im = cv2.imread('8.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#Nhị phân hoá ảnh với THRESH_BINARY_INV để chuyển đối tượng có màu đen nền trắng sang ngược lại
_, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)
#Tìm countour của ảnh - contour chỉ được xác định trên đối tượng có màu trắng nền đen
contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
boxes = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    boxes.append([x, 0, x + w, im.shape[0]])
    cv2.rectangle(im, (x, 0), (x + w, y + h), (255, 0, 0), 1)

# cv2.drawContours(im, contours, -1, (255, 0, 0), 2)

cv2.imshow('cc', thresh)
cv2.imshow('vv', im)
cv2.waitKey()

# image = cv2.imread( "images/9.png")
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# th, threshed = cv2.threshold(gray_image, 100, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
# cnts = cv2.findContours(cv2.morphologyEx(threshed, cv2.MORPH_OPEN, np.ones((2,2))), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
# nh, nw = image.shape[:2]
# for cnt in cnts:
#     x,y,w,h = bbox = cv2.boundingRect(cnt)
#     if h >= 0.3 * nh:
#         cv2.rectangle(image, (x,y), (x+w, y+h), (0, 0, 0), 1, cv2.LINE_AA)
# cv2.imshow("dst", image)
# cv2.waitKey()