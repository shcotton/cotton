import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys

def f(x):
    R, G, B = x
    RB = abs(R-B)/(R+B)
    RG = abs(R-G)/(R+G)
    BG = abs(B-G)/(B+G)
    return abs(RB-RG), abs(RG-BG), abs(BG-RB)

img = sys.argv[1]
lbl = sys.argv[2]
img = cv2.imread(img)
lbl = cv2.imread(lbl, 0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float)+1
img2 = img.reshape(-1, 3)
img2 = np.apply_along_axis(f, 1, img2)
lbl2 = lbl.reshape(-1)
x,y,z = img2[:,0], img2[:,1], img2[:,2]
c = np.random.choice(img2.shape[0], size=10000, replace=False)
x,y,z = x[c], y[c], z[c]
lbl2 = lbl2[c]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ff = lambda x: 'red' if x else 'blue'
colormap = np.vectorize(ff)(lbl2)

ax.scatter(x,y,z, marker='o', c=colormap)
ax.set_xlabel('L Label')
ax.set_ylabel('A Label')
ax.set_zlabel('B Label')
plt.show()
