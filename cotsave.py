import sys
import cv2
import numpy as np

filename = sys.argv[1]
mask = np.load(filename)
unique_elements, counts_elements = np.unique(mask, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))
newfile = filename.rsplit('.', maxsplit=1)[0] + '_mask.jpg'

cv2.imwrite(newfile, mask)

