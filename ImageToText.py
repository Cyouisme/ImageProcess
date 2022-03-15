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


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB):
    # compute the mean squared error and structural similarity
    # index for the images
    result = mse(imageA, imageB)
    return result

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


# a = cv2.imread('images/a.png')
# idx = np.where(np.all(a == (34, 34, 34), 2))
# if not len(idx[0]) == 0 and not len(idx[1]) == 0:
#     x1, y1, x2, y2 = idx[1].min(), idx[0].min(), idx[1].max(), idx[0].max()
#     a = a[y1:y2, x1:x2]
# a = convert_image(a)
# b = cv2.imread('images/b.png')
#
# # re-block the figure
# #np.where just contain two values: True or False
# #Run all numpy arr and compare value((34, 34, 34)) and covert True-False
# idx = np.where(np.all(b == (34, 34, 34), 2))
# if not len(idx[0]) == 0 and not len(idx[1]) == 0:
#     x1, y1, x2, y2 = idx[1].min(), idx[0].min(), idx[1].max(), idx[0].max()
#     b = b[y1:y2, x1:x2]
#
# # convert numpy array to image
# b = convert_image(b)
# # c = Image.resresize(a, (16, 50), interpolation=cv2.INTER_AREA)
# a = a.resize((16, 50))
# b = b.resize((16, 50))
# a.show()
# b.show()
# # cv2.imshow('b',b)
# print(equal(a, b))
# cv2.waitKey()

imgDict = {}
# đọc hết tất cả các ảnh từ file
for p in Path('images/data').rglob('*.[jp][pn]*'):
    # try:
    im = Image.open(p.as_posix())#.convert('L')  # cv2.imread(p.as_posix(), 0)
    # convert image to np arr
    im = np.array(im)
    labels = p.name.split('.')[0].split(' ')
    for label in labels:
        if label == "":
            labels.remove(label)
    # chuyển ảnh thành nền đen chữ trắng
    _, thresh = cv2.threshold(cv2.cvtColor(im.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # cnts = contours.sort_contours(cnts,method="left-to-right")[0]
    boxes = []
    # enumerate: thay thế i=0,i++
    for (i, c) in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(c)
        if w > 30 or h > 15:
            boxes.append([x, 0, x + w, im.shape[0]])
    boxes = sorted(boxes, key=lambda x: x[0])
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        crop_i = im[y1:y2, x1:x2]
        # idx = np.where(np.all(crop == (34, 34, 34), 2))
        # if not len(idx[0]) == 0 and not len(idx[1]) == 0:
        #     x1, y1, x2, y2 = idx[1].min(), idx[0].min(), idx[1].max(), idx[0].max()
        #     # print(x1, y1, x2, y2)
        #     # cv2.rectangle(crop, (x1, y1), (x2, y2), (0, 0, 0), 2)
        Image.fromarray(crop_i).save(f'images/imageF/{labels[i]}.png')
        imgDict.update({labels[i]: crop_i})
        # print(labels[i])
        # cv2.imshow('ff', crop)
        # cv2.waitKey()
    # except:
    #     cv2.imshow('tt', im)
    #     cv2.waitKey()


im = cv2.imread('images/imageInput/9.png')
imgray = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2GRAY)
cv2.imshow('imgray', imgray)
kernel = np.ones((2, 2), np.uint8)
dilation = cv2.dilate(imgray.copy(), kernel, iterations=1)
cv2.imshow('dilation', dilation)
# Nhị phân hoá ảnh với THRESH_BINARY_INV để chuyển đối tượng có màu đen nền trắng sang ngược lại
_, thresh = cv2.threshold(dilation.copy(), 127, 255, cv2.THRESH_BINARY_INV)
# Tìm countour của ảnh - contour chỉ được xác định trên đối tượng có màu trắng nền đen
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('cc', thresh)

boxes = []
for (i, c) in enumerate(contours):
    x, y, w, h = cv2.boundingRect(c)
    # cv2.rectangle(im, (x, y), (x + w), (255, 0, 0), 1)
    if w > 5 or h > 2:
        boxes.append([x, 0, x + w, im.shape[0]])
boxes = sorted(boxes, key=lambda x: x[0])
for i, (x1, y1, x2, y2) in enumerate(boxes):
    crop = im[y1:y2, x1:x2]
    idx = np.where(np.all(crop == (34, 34, 34), axis=2))
    if not len(idx[0]) == 0 and not len(idx[1]) == 0:
        x1, y1, x2, y2 = idx[1].min(), idx[0].min(), idx[1].max(), idx[0].max()
        # cv2.rectangle(value, (x1, y1), (x2, y2), (0, 0, 0), 2)
        crop = crop[y1:y2, x1:x2]
    crop = Image.fromarray(np.array(crop)).resize((16,50))
    # crop.show()
    for value in imgDict.values(): # đây nè, t lặp một vòng lặp trong cái imgDict để lấy cái value của hắn ra
        idx = np.where(np.all(value == (34, 34, 34), axis=2))
        if not len(idx[0]) == 0 and not len(idx[1]) == 0:
            x1, y1, x2, y2 = idx[1].min(), idx[0].min(), idx[1].max(), idx[0].max()
            # cv2.rectangle(value, (x1, y1), (x2, y2), (0, 0, 0), 2)
            value = value[y1:y2, x1:x2]
        value = Image.fromarray(value)
        value = value.resize((16,50))
        # value.show()
        if equal(value, crop) is None == False:
            print(get_key(value))
        else:
            continue
    # for j in range(30):
    #     Image.fromarray(crop).save(f'images/imageOutput/{random.random()}.png')

# for f in Path('images/imageF').rglob('*.[jp][pn]*'):
#     img = Image.open(f.as_posix()).convert('L')
#     labels = f.name.split('.')[0].split(' ')
#     for d in Path('images/imageOutput').rglob('*.[jp][pn]*'):
#         img1 = Image.open(d.as_posix()).convert('L')
#         if equal(img, img1) is None:
#             print(labels)

# for d in Path('images/imageOutput').rglob('*.[jp][pn]*'):
#     img_compare = Image.open(d.as_posix()).convert('L')
#     for value in imgDict.values():
#         idx = np.where(np.all(value == (34, 34, 34), 2))
#         if not len(idx[0]) == 0 and not len(idx[1]) == 0:
#             x1, y1, x2, y2 = idx[1].min(), idx[0].min(), idx[1].max(), idx[0].max()
#             print(x1, y1, x2, y2)
#             cv2.rectangle(value, (x1, y1), (x2, y2), (0, 0, 0), 2)
#         if equal(convert_image(value), img_compare) is None:
#             print(get_key(value))

# for c in contours:
#     x, y, w, h = cv2.boundingRect(c)
#     boxes.append([x, 0, x + w, im.shape[0]])
#     cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 1)
# boxes = sorted(boxes, key=lambda x: x[0])
# crop = im[boxes[0][1]:boxes[0][3], boxes[0][0]:boxes[0][2]]
#
# # for i in len(boxes):
# #     if equal(crop,imgDict[])
# # cv2.drawContours(im, contours, -1, (255, 0, 0), 2)
#
# cv2.imshow('vv', im)
cv2.waitKey()
