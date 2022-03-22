import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from input_text_detection import read_dataset
import time


start_time = time.time()


def overlapping_image(size1, size2):
    (x1, y1, w1, h1), image1 = size1
    (x2, y2, w2, h2), image2 = size2
    if x1 + w1 > x2 and x2 + w2 > x1 and y1 + h1 > y2 and y2 + h2 > y1:
        minx = min(x1, x2)
        miny = min(y1, y2)
        mask = np.zeros((max(y1 + h1, y2 + h2) - miny, max(x1 + w1, x2 + w2) - minx), dtype=np.uint8)
        mask[y1 - miny:y1 - miny + h1, x1 - minx:x1 - minx + w1] = image1
        mask[y2 - miny:y2 - miny + h2, x2 - minx:x2 - minx + w2] = image2
        return True, mask
    return False, image1


imgDict = {}
dataset = read_dataset("images/data")


def image_to_text():
    #Read image form file and convert to gray image
    for f in Path('images/imageInput').rglob('*.[jp][pn]*'):
        im = Image.open(f.as_posix()).convert('L')
        #Convert image to numpy arr
        im = np.array(im)
        # im = cv2.GaussianBlur(im, (3,3), cv2.BORDER_DEFAULT)
        # im = cv2.blur(im, (3,3))
        thresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)[1]
        dilated = cv2.dilate(thresh.copy(), np.ones((5, 1), np.uint8), iterations=1)
        contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dir_images = {}
        list_images = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if (h, w) != dilated.shape[:2] and h >= 2:
                dir_images.setdefault(x, ((x, y, w, h), thresh[y:y + h, x:x + w]))
            # cv2.rectangle(im, (x, y), (x + w, y + h), (128, 128, 128), 2)
        list_img = sorted(dir_images.items(), reverse=False)[:-1]
        i = 0
        while i <= len(list_img) - 2:
            check, img = overlapping_image(list_img[i][1], list_img[i + 1][1])
            #Compare the end point of the previous word with the first point of the following word
            nx, ny, nw, nh = list_img[i + 1][1][0]
            #Add image to list with image fixed and check 0-1 for space
            list_images.append(
                [cv2.copyMakeBorder(cv2.resize(img, (24, 43)), 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(0, 0, 0)),
                 1 if (nx - (list_img[i][1][0][0] + list_img[i][1][0][2])) > 7 else 0])
            if check:
                i += 1
            if i == len(list_img) - 2:
                list_images.append(
                    [cv2.copyMakeBorder(cv2.resize(list_img[len(list_img) - 1][1][1], (24, 43)), 30, 30, 30, 30,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0)),
                     1 if (nx - (list_img[i][1][0][0] + list_img[i][1][0][2])) > 7 else 0])
            i += 1
        # Compare input and output image and print it to screen
        result = ''
        for i, (image, isSpace) in enumerate(list_images):
            min_diff = []
            for image_data in dataset:
                #Compare two images with number pixel different
                #Stack two image and count white point -> different point
                #The more similar two images are, the less white points they have
                res = cv2.absdiff(image, image_data[1])
                diff_count = cv2.countNonZero(res)
                if len(min_diff) == 0:
                    min_diff = [image_data[0], diff_count]
                else:
                    if min_diff[1] > diff_count:
                        min_diff = [image_data[0], diff_count]
            #Check space in sentence and print sentences from image to screen
            if isSpace == 1:
                result += min_diff[0] + ' '
            else:
                result += min_diff[0]
        print(result)


image_to_text()
#Count time run python execute
print("--- %s seconds ---" % (time.time() - start_time))
cv2.waitKey()
