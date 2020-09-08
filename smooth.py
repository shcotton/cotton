import numpy as np
import cv2
import sys

image_path = sys.argv[1]
output_path = sys.argv[2]

mask = cv2.imread(image_path, 0)

cnts = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 600:
        cv2.drawContours(mask, [c], -1, 0, -1)

k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k2)

cv2.imwrite(output_path, mask)
