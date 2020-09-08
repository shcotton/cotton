import cv2
import numpy as np
from glob import glob
import argparse
import os
import pickle as pkl
import features_full as ft

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
    features, _ = ft.get_features_labels(image, None, train=False)
    return features, (image.shape[0], image.shape[1])

def infer_model(model, X):
    y = model.predict(X)
    return y

def write_data(y, shape, output):
    mask = y.reshape(*shape)
    cv2.imwrite(output, mask)

def main(image, model, output):
    X, shape = read_data(image)
    y = infer_model(pkl.load(open(model,"rb")), X)
    write_data(y, shape, output)

if __name__ == '__main__':
    args = parse_args()
    image = args.image
    model = args.model
    output = args.output
    main(image, model, output)
