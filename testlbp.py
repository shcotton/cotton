import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import sys

def stat(a):
    unique_elements, counts_elements = np.unique(a, return_counts=True)
    return np.asarray((unique_elements, counts_elements))

img_path = sys.argv[1]
img = cv2.imread(img_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

lbp_image = local_binary_pattern(img_gray, 8, 2, method='nri_uniform')
print(lbp_image)
print(lbp_image.shape)
print(lbp_image.min())
