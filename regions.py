import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.segmentation import slic, mark_boundaries
import sys

def to_regions(img, p=100):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    n_seg = img.shape[0] * img.shape[1] // p
    seg_slic = slic(img, n_segments=n_seg, start_label=0)
    return seg_slic

def get_regions(img, regs, raw=True):
    if raw:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if len(img.shape) > 2:
        img = img.reshape(-1, 3)
    else:
        img = img.ravel()
    regs = regs.ravel()
    mu = np.unique(regs)
    m = [[] for i in range(len(mu))]
    for r, i in zip(regs, img):
        m[r].append(i)
    m = [np.array(l) for l in m]
    return m

def get_indices(regs):
    return np.unique(regs, return_index=True)[1]

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    imr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    seg_slic = to_regions(img, 800)
    print('start')
    p = get_regions(img, seg_slic)
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
