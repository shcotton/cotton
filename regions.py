import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.segmentation import slic, mark_boundaries
import networkx as nx
import sys

class Region:
    def __init__(self, pts):
        self.mean = np.mean(pts, axis=0)
        self.cov = np.cov(pts, rowvar=False)

def reg_dist(X, Y):
    a = np.sum((X.mean - Y.mean) ** 2)
    b = np.trace(X.cov) + np.trace(Y.cov)
    c = np.sum(np.sqrt(np.linalg.eig(np.dot(X.cov, Y.cov))[0]))
    return np.sqrt(a + b - 2 * c)

def get_graph(regs):
    vs_right = np.vstack([regs[:,:-1].ravel(), regs[:,1:].ravel()])
    vs_below = np.vstack([regs[:-1,:].ravel(), regs[1:,:].ravel()])
    x = np.unique(np.hstack([vs_right, vs_below]).T, axis=0)
    x.sort(axis=1)
    x = np.unique(x, axis=0)
    x = x[x[:,0] != x[:,1]]
    G = nx.Graph()
    G.add_edges_from(x)
    return G

def dbscan(img, regs, raw=False):
    if raw:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    G = get_graph(regs)
    props = regionprops(regs + 1)
    n = len(props)
    C = 0
    rs = [Region(img[tuple(prop.coords.T)]) for prop in props]
    print(reg_dist(rs[10], rs[5]))

    visit = np.zeros(n, dtype=bool)
    lbl = np.zeros(n, dtype=np.uint32)
    for i in range(n):
        if visit[i]:
            continue
        visit[i] = True
        nbs = G.neighbors(i)
        nbs = filter(lambda x: reg_dist(rs[i], rs[x]) < 13, nbs)
        nbs = list(nbs)
        lbl[i] = C
        j = 0
        while j < len(nbs):
            nb = nbs[j]
            if not visit[nb]:
                visit[nb] = True
                lbl[nb] = C
                nbs2 = G.neighbors(nb)
                nbs2 = filter(lambda x: reg_dist(rs[nb], rs[x]) < 13, nbs2)
                nbs.extend(nbs2)
            j += 1
        C += 1
    regs = lbl[regs]
    return regs

def to_regions(img, p=100):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    n_seg = img.shape[0] * img.shape[1] // p
    seg_slic = slic(img, n_segments=n_seg)
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
    # p = get_regions(img, seg_slic)
    regs = dbscan(img, seg_slic, raw=True)
    print(regs)
    print('end')
    bb = mark_boundaries(imr, regs)
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
