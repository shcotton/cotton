import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np

img = sys.argv[1]
maskname = img.rsplit('.', maxsplit=1)[0] + '.npy'
cot = cv2.imread(img)
cot = cv2.cvtColor(cot, cv2.COLOR_BGR2RGB)
hsv_cot = cv2.cvtColor(cot, cv2.COLOR_RGB2HSV)

light = (0, 0, 100)
dark = (255, 60, 255)
mask = cv2.inRange(hsv_cot, light, dark)
se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
np.save(maskname, mask)
#result = cv2.bitwise_and(cot, cot, mask=mask)
#mask = cv2.fastNlMeansDenoising(mask)
plt.subplot(1, 2, 1)
plt.imshow(cot)
plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.show()
