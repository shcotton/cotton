import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import time

ohta = np.array([
    [ 1/3, 1/3, 1/3 ],
    [ 1/2, 0., -1/2 ],
    [ -1/4, 1/2, -1/4 ]
])

def smooth(mask):
    cnts = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 200:
            cv2.drawContours(mask, [c], -1, 0, -1)

img_orig = cv2.imread(sys.argv[1])
img_orig = cv2.fastNlMeansDenoisingColored(img_orig,None,10,10,7,21)

img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_orig, cv2.COLOR_BGR2HSV)
img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
img = cv2.imread(sys.argv[1], 0)
img = cv2.medianBlur(img, 5)

img_rgb2 = img_rgb.astype(int)
img_g = img_rgb2[:,:,1] * 2 - img_rgb2[:,:,0] - img_rgb2[:,:,2]
img_g = img_g > 110
img_g = img_g.astype(np.uint8)
img_g *= 255

img_r = img_rgb2[:,:,0] * 2 - img_rgb2[:,:,1] - img_rgb2[:,:,2]
img_r = img_r > 30
img_r = img_r.astype(np.uint8)
img_r *= 255

img_b = img_rgb2[:,:,2] * 2 - img_rgb2[:,:,0] - img_rgb2[:,:,1]
img_b = img_b > 30
img_b = img_b.astype(np.uint8)
img_b *= 255
# global thresholding
# x = time.time()
ret3,th3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
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
mask_white = cv2.inRange(img_hsv, (0, 0, 175), (179, 50, 255))
mask_green = cv2.inRange(img_hsv, (40, 50, 0), (100, 255, 170))
mask_brown = cv2.inRange(img_hsv, (10, 60, 0), (30, 255, 150))
mask_bb = cv2.inRange(img_gray, 200, 255)
mask = cv2.bitwise_or(mask_green, mask_brown)
# mask = cv2.bitwise_or(mask, img_r)
# mask = cv2.bitwise_or(mask, img_g)
# mask = cv2.bitwise_or(mask, img_b)
mask = cv2.bitwise_not(mask)
mask = cv2.bitwise_and(mask, mask_white)
# mask = cv2.bitwise_and(mask, th3)
# mask = cv2.bitwise_and(mask, mask_bb)
# smooth(mask)
y = time.time() - x
print(y)

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.subplot(1, 3, 2)
plt.imshow(img_hsv)
plt.subplot(1, 3, 3)
plt.imshow(mask, 'gray')
plt.show()

