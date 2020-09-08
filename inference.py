import cv2
import numpy as np
from glob import glob
import argparse
import os
import pickle as pkl
import features as ft

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
    regions = ft.get_regions(image, 800)
    features = ft.get_features(image, regions)
    return features, regions

def infer_model(model, X):
    y = model.predict(X)
    return y

def write_data(y, regions, output):
    inference_img = y[regions]
    cv2.imwrite(output, inference_img)

def main(image, model, output):
    X, regions = read_data(image)
    y = infer_model(pkl.load(open(model,"rb")), X)
    write_data(y, regions, output)

if __name__ == '__main__':
    args = parse_args()
    image = args.image
    model = args.model
    output = args.output
    main(image, model, output)
