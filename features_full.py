import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import sys

ohta = np.array([
    [ 1/3, 1/3, 1/3 ],
    [ -1/2, 0., 1/2 ],
    [ -1/4, 1/2, -1/4 ]
])

def subsample_idx(low, high, sample_size):
    return np.random.randint(low, high, sample_size)

def create_binary_pattern(img, p, r):
    #print ('[INFO] Computing local binary pattern features.')
    lbp = local_binary_pattern(img, p, r)
    return (lbp-np.min(lbp))/(np.max(lbp)-np.min(lbp)) * 255

def get_features_labels(img, label, train=True):
    lbp_radius = 24 # local binary pattern neighbourhood
    h_neigh = 11 # haralick neighbourhood
    num_examples = 10000 # number of examples per image to use for training model

    lbp_points = lbp_radius*8
    h_ind = (h_neigh - 1) // 2    # 5

    feature_img = np.zeros((img.shape[0], img.shape[1], 4))
    feature_img[:,:,:3] = img.dot(ohta.T)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feature_img[:,:,3] = create_binary_pattern(img_gray, lbp_points, lbp_radius)
    features = feature_img.reshape(feature_img.shape[0]*feature_img.shape[1], feature_img.shape[2])
    if train == True:
        ss_idx = subsample_idx(0, features.shape[0], num_examples)
        features = features[ss_idx]
    else:
        ss_idx = []
    if train == True:
        labels = label.ravel()[ss_idx]
    else:
        labels = None
    return features, labels

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
