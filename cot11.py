import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

img = sys.argv[1]
cot = cv2.imread(img)
cot = cv2.cvtColor(cot, cv2.COLOR_BGR2RGB)
hsv_cot = cv2.cvtColor(cot, cv2.COLOR_RGB2HSV)

light = (90, 0, 100)
dark = (179, 80, 255)
x = time.time()
mask = cv2.inRange(hsv_cot, light, dark)
time.sleep(1)
y = time.time()
print(y-x,np.unique(mask))

cnts = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 1000:
        cv2.drawContours(mask, [c], -1, 0, -1)

#se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
#mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
#result = cv2.bitwise_and(cot, cot, mask=mask)
#mask = cv2.fastNlMeansDenoising(mask)
plt.subplot(1, 2, 1)
plt.imshow(cot)
plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.show()
