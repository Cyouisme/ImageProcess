import imutils
import numpy as np
import cv2
from pathlib import Path
from PIL import ImageChops, Image
import random
from imutils import contours


def equal(im1, im2):
    return ImageChops.difference(im1, im2)


a = cv2.imread('images/imageF/Ế.png')
a = Image.open()
# idx = np.where(np.all(a == (34, 34, 34), 2))
# if not len(idx[0]) == 0 and not len(idx[1]) == 0:
#     x1, y1, x2, y2 = idx[1].min(), idx[0].min(), idx[1].max(), idx[0].max()
#     a = a[y1:y2, x1:x2]
a = Image.fromarray(a)
b = cv2.imread('images/imageOutput/0.17366320358374066.png')


# re-block the figure
# np.where just contain two values: True or False
# Run all numpy arr and compare value((34, 34, 34)) and covert True-False
# idx = np.where(np.all(b == (34, 34, 34), 2))
# if not len(idx[0]) == 0 and not len(idx[1]) == 0:
#     x1, y1, x2, y2 = idx[1].min(), idx[0].min(), idx[1].max(), idx[0].max()
#     b = b[y1:y2, x1:x2]

# convert numpy array to image
b = Image.fromarray(b)
# c = Image.resresize(a, (16, 50), interpolation=cv2.INTER_AREA)
a = a.resize((16, 50))
b = b.resize((16, 50))
diff = equal(a,b)
w, h = diff.size
score = 0
for x in range(w):
    for y in range(h):
        score += max(diff.getpixel((x,y)))
if score < 10500:
    print('similar')
# cv2.rectangle(np.array(b), (x1,y1), (x2,y2), (255,0,0),2)
# b = Image.fromarray(np.array(b))
# a.show()
# b.show()
# cv2.imshow('b',b)
# print(equal(a, b))

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
