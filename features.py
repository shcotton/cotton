import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage.measure import regionprops
import sys
import regions as rg

# def get_mean(g):
#     res = np.zeros((len(g), 3), dtype=np.uint8)
#     for i, a in enumerate(g):
#         res[i, :] = a.mean(axis=0)
#     return res

def get_mean(r):
    return r.mean(axis=0)

def get_hist(r):
    def f(x):
        R, G, B = x
        # RB = abs(R-B)/(R+B)
        # RG = abs(R-G)/(R+G)
        # BG = abs(B-G)/(B+G)
        return abs(R-G), abs(G-B), abs(B-R)

    r = r.astype(np.int32)
    nr = np.apply_along_axis(f, 1, r)

    res = np.zeros(48)
    for i in range(3):
        res[i<<4 : (i+1)<<4] = np.histogram(nr[:,i], bins=16, range=(0, 256))[0] / nr.shape[0]
    return res

def get_lbp(r):
    return np.bincount(r, minlength=59) / len(r)

def get_lbp_regions(img, regs):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imglbp = local_binary_pattern(img, 8, 2, method='nri_uniform').astype(np.uint32)
    g = rg.get_regions(imglbp, regs, raw=False)
    return g

def get_regions(img, p=100):
    return rg.to_regions(img, p)

def get_features(img, regs):
    g = rg.get_regions(img, regs)
    res = np.zeros((len(g), 110))
    for i, r in enumerate(g):
        res[i, :3] = get_mean(r)
        res[i, 3:51] = get_hist(r)

    lg = get_lbp_regions(img, regs)
    for i, r in enumerate(lg):
        res[i, 51:] = get_lbp(r)
    return res

def get_labels(label, regs):
    props = regionprops(regs + 1)
    labels = np.zeros(len(props), dtype=np.uint8)
    for reg, prop in enumerate(props):
        x, y = prop.centroid
        cent = int(x), int(y)
        labels[reg] = label[cent]
    return labels

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    label = cv2.imread(sys.argv[1], 0)
    print('start')
    np.set_printoptions(threshold=sys.maxsize)
    regs = get_regions(img, 800)
    f = get_features(img, regs)
    l = get_labels(label, regs)
    print(f)
    print(l)
    print(f.shape, l.shape)
