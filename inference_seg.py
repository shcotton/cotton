import cv2
import numpy as np
from glob import glob
import argparse
import os
import pickle as pkl
import features as ft
import features_full as ftf
from skimage.measure import regionprops
import time

def check_args(args):
    if not os.path.exists(args.image):
        raise ValueError("Image file does not exist")
    return args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image" , help="Path to image", required=True)
    parser.add_argument("-m", "--model", help="Path to model", required=True)
    parser.add_argument("-o", "--output", help="Path to output image", required=True)
    args = parser.parse_args()
    return check_args(args)

def read_data(image):
    image = cv2.imread(image)
    # image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    regions = ft.get_regions(image, 800)
    return image, regions

def feature_and_infer(model, image, regions):
    p = time.time()
    rs = regionprops(regions + 1)
    f, _ = ftf.get_features_labels(image, None, train=False, reshape=False)
    results = np.zeros(len(rs), dtype=np.uint8)
    for i, r in enumerate(rs):
        if r.area < 300:
            results[i] = 0
            continue
        coords = r.coords
        pop = len(coords)
        if pop > 50:
            choices = np.random.choice(pop, size=50, replace=False)
            choices = coords[choices]
            pop = 50
        else:
            choices = coords
        # print(f.shape)
        # print(choices.shape)
        X = f[tuple(choices.T)]
        # print(X.shape)
        y = model.predict(X)
        t = np.count_nonzero(y)
        if t / pop > 0.4:
            results[i] = 255
        else:
            results[i] = 0
    p2 = time.time()
    print(p2 - p)
    mask = results[regions]
    print(mask)
    return mask

def write_data(output, mask):
    cv2.imwrite(output, mask)

def main(image, model, output):
    image, regions = read_data(image)
    model = pkl.load(open(model, "rb"))
    mask = feature_and_infer(model, image, regions)
    write_data(output, mask)

if __name__ == '__main__':
    args = parse_args()
    image = args.image
    model = args.model
    output = args.output
    main(image, model, output)
