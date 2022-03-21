import cv2
import imutils
import numpy as np
import math
from imutils import contours
from scipy.spatial import distance as dist
from imutils import perspective


def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


list_cnts = []
im = cv2.imread('images/imageInput/8.png')
# define x_max and y_max
img_h, img_w = im.shape[:2]
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)[1]
# dilated = cv2.dilate(thresh.copy(), np.ones((5, 1), np.uint8), iterations=1)
# cv2.imshow('im', im)
# cv2.imshow('t', dilated)
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
boxes = []
space = []
# img[2][2]
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    boxes.append([x, 0, x+w, img_h])
boxes = sorted(boxes, key=lambda x: x[0])
for i, (x1, y1, x2, y2) in enumerate(boxes):
    if i > len(boxes):
        nx, ny, nw, nh = boxes[i+1]
        if (nx - x2) > 15:
            space.append(1)
    else:
        print('Fail!')
print(space)
#     crop = thresh[y1:y2, x1:x2]
#     cv2.imshow('cr', crop)
#     print(x, y)
#     list_cnts.append([x, y])
    # M = cv2.moments(c)
    # if M['m00'] != 0:
    #     cx = int(M['m10'] / M['m00'])
    #     cy = int(M['m01'] / M['m00'])
#     # print(center)
#     cv2.rectangle(im, (x, y), (x +w, y+h), (0, 255, 0), 2)
    # cv2.line(img=im, pt1=(x, 0), pt2=(x, h), color=(0, 0, 0),
    #          thickness=1, lineType=cv2.LINE_AA, shift=0)

# for i in range(len(list_cnts)-1):
#     d = calculateDistance(list_cnts[i][0],list_cnts[i][1],list_cnts[i+1][0],list_cnts[i+1][1])
#     print(d, end=' ')
# cv2.imshow('im', im)

cv2.waitKey()

