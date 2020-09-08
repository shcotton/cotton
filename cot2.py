import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

img = sys.argv[1]
K = int(sys.argv[2])
cot = cv2.imread(img)
cot = cv2.cvtColor(cot, cv2.COLOR_BGR2RGB)
vec_cot = cot.reshape((-1, 3))
vec_cot = np.float32(vec_cot)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
attempts = 10
ret, label, center = cv2.kmeans(vec_cot, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((cot.shape))

plt.subplot(1, 2, 1)
plt.imshow(cot)
plt.subplot(1, 2, 2)
plt.imshow(result_image)
plt.show()
