import cv2
import numpy as np
import sys

fin = sys.argv[1]
fout = sys.argv[2]

img = cv2.imread(fin, 0)
y = np.zeros(img.shape, dtype=np.uint8)
cv2.imwrite(fout, y)
