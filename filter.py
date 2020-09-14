import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import time

def smooth(mask):
    cnts = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 500:
            cv2.drawContours(mask, [c], -1, 0, -1)

img_orig = cv2.imread(sys.argv[1])
img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_orig, cv2.COLOR_BGR2HSV)
img_lab = cv2.cvtColor(img_orig, cv2.COLOR_BGR2LAB)
img = cv2.imread(sys.argv[1], 0)
img = cv2.medianBlur(img, 5)
# global thresholding
# x = time.time()
ret3,th3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(ret3)
# ret, th3 = cv2.threshold(img_hsv[:,:,1], 100, 255, cv2.THRESH_BINARY)
# y = time.time() - x
# print(y)

kernel = np.ones((5,5),np.uint8)
# th3 = cv2.dilate(th3, kernel, iterations = 1)
# # smooth(th3)
# # plot all the images and their histograms
x = time.time()
# mask_yellow = cv2.inRange(img_hsv, (21, 39, 64), (40, 255, 255))
# mask_white = cv2.inRange(img_hsv, (0, 0, 80), (179, 150, 255))
mask_white = cv2.inRange(img_hsv, (0, 0, 80), (179, 100, 255))
mask_green = cv2.inRange(img_hsv, (40, 50, 0), (100, 255, 170))
mask_brown = cv2.inRange(img_hsv, (10, 60, 0), (30, 255, 150))
mask = cv2.bitwise_or(mask_green, mask_brown)
mask = cv2.bitwise_not(mask)
mask = cv2.bitwise_and(mask, mask_white)
smooth(mask)
y = time.time() - x
print(y)

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.subplot(1, 3, 2)
plt.imshow(img_hsv)
plt.subplot(1, 3, 3)
plt.imshow(mask, 'gray')
plt.show()

