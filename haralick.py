import sys, cv2
import numpy as np
from skimage import io
from scipy import stats
from skimage.feature import greycoprops
import mahotas as mt

def extract_features(image):
        # calculate haralick texture features for 4 types of adjacency

        textures = mt.features.haralick(image)
        #print textures

        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)

        #std = StandardScaler().fit(ht_mean)
        #ht_mean = std.transform(ht_mean)

        #sift = cv2.xfeatures2d.SIFT_create()
        #kp = sift.detect(image,None)

        #features = cv2.FeatureDetector_create("SIFT")
        #desc = cv2.DescriptorExtractor_create("SIFT")

        #print ht_mean

        return ht_mean
        #return kp

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1], 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(extract_features(img))
