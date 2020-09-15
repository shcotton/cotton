import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.segmentation import slic, felzenszwalb, mark_boundaries
import sys

def to_regions(img, p=100):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    n_seg = img.shape[0] * img.shape[1] // p
    seg_slic = slic(img, n_segments=n_seg)
    #seg_slic = felzenszwalb(img)
    return seg_slic

def enc_regions(regs):
    return np.cumsum(np.unique(regs.ravel(), return_counts=True)[1])

def get_regions(img, mreg, raw=True):
    if raw:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if len(img.shape) > 2:
        img = img.reshape(-1, 3)
    else:
        img = img.ravel()
    s = np.split(img, mreg)
    s.pop()
    return s

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    imr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    seg_slic = to_regions(img, 500)
    print('start')
    r = regionprops(seg_slic + 1, intensity_image=imr[:,:,1])
    print('end')
    bb = mark_boundaries(imr, seg_slic)
    plt.imshow(bb)
    plt.show()

# print('2')
# sr = seg_slic.ravel()
# ir = image.reshape(-1, 3)
# df = np.c_[ir, sr]
# df = pd.DataFrame(df).groupby(3)
# m = df.mean().to_numpy()
# c = df.cov().to_numpy().reshape(-1, 3, 3)
# print('3')
# print(m)
# print(c)
# r = regionprops(seg_slic + 1, intensity_image=image[:,:,1])
# print(len(r))

#np.dstack([regionprops(seg_slic, intensity_image=image[:,:,i]) for i in range(3)])

# print(regions)
# regions = regionprops(segments_slic, intensity_image=rgb2gray(image))
# for props in regions:
#     cy, cx = props.centroid
#     plt.plot(cx, cy, 'ro')
# print(stat(segments_slic))
