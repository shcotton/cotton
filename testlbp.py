import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import sys
import matplotlib.pyplot as plt

def stat(a):
    unique_elements, counts_elements = np.unique(a, return_counts=True)
    return np.asarray((unique_elements, counts_elements))

img_path = sys.argv[1]
img = cv2.imread(img_path)
imgg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

r = 5
lbp = local_binary_pattern(img_gray, 8, 2, method='nri_uniform')
plt.subplot(121)
plt.imshow(imgg)
plt.subplot(122)
plt.imshow(lbp, 'gray')
plt.show()
