import cv2
import numpy as np
from glob import glob
import os
import argparse
import pickle as pkl
from sklearn import metrics
from sklearn.model_selection import train_test_split
import features_full as ft

def check_args(args):
    if not os.path.exists(args.image_dir) or not os.path.exists(args.label_dir):
        raise ValueError("Directory does not exist")
    if args.classifier != "SVM" and args.classifier != "RF" and args.classifier != "GBC":
        raise ValueError("Classifier must be either SVM, RF or GBC")
    return args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_dir" , help="Path to images", required=True)
    parser.add_argument("-l", "--label_dir" , help="Path to labels", required=True)
    parser.add_argument("-c", "--classifier", help="Classification model to use", required=True)
    parser.add_argument("-o", "--output_model", help="Path to save model", required=True)
    args = parser.parse_args()
    return check_args(args)

def read_data(image_dir, label_dir):
    image_files = glob(os.path.join(image_dir, '*.jpg'))
    label_files = glob(os.path.join(label_dir, '*.jpg'))
    image_files.sort()
    label_files.sort()
    image_list = []
    label_list = []

    for image_file, label_file in zip(image_files, label_files):
        image_list.append((image_file, cv2.imread(image_file, 1)))
        label_list.append((label_file, cv2.imread(label_file, 0)))

    return image_list, label_list

def create_training_dataset(image_list, label_list):
    print ('[INFO] Creating training dataset on %d image(s).' %len(image_list))

    X = []
    y = []

    for i, (image, label) in enumerate(zip(image_list, label_list)):
        image_file, image = image
        label_file, label = label
        image_name = os.path.basename(image_file)
        print(f'Now on {image_name}')
        features, labels = ft.get_features_labels(image, label)
        X.append(features)
        y.append(labels)

    X = np.vstack(X)
    y = np.concatenate(y)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print ('[INFO] Feature vector size:', X.shape)

    return X, y

def train_model(X, y, classifier="RF"):
    if classifier == "SVM":
        from sklearn.svm import SVC
        print ('[INFO] Training Support Vector Machine model.')
        model = SVC(C=10., kernel='linear')
        model.fit(X, y)
    elif classifier == "RF":
        from sklearn.ensemble import RandomForestClassifier
        print ('[INFO] Training Random Forest model.')
        model = RandomForestClassifier(n_estimators=250, max_depth=12, random_state=42)
        model.fit(X, y)
    elif classifier == "GBC":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        model.fit(X, y)

    print ('[INFO] Model training complete.')
    print ('[INFO] Training Accuracy: %.2f' % model.score(X, y))
    return model

def test_model(X, y, model):

    pred = model.predict(X)
    precision = metrics.precision_score(y, pred, average='weighted', labels=np.unique(pred))
    recall = metrics.recall_score(y, pred, average='weighted', labels=np.unique(pred))
    f1 = metrics.f1_score(y, pred, average='weighted', labels=np.unique(pred))
    accuracy = metrics.accuracy_score(y, pred)

    print ('--------------------------------')
    print ('[RESULTS] Accuracy: %.2f' %accuracy)
    print ('[RESULTS] Precision: %.2f' %precision)
    print ('[RESULTS] Recall: %.2f' %recall)
    print ('[RESULTS] F1: %.2f' %f1)
    print ('--------------------------------')

def main(image_dir, label_dir, classifier, output_model):
    image_list, label_list = read_data(image_dir, label_dir)
    X, y = create_training_dataset(image_list, label_list)
    model = train_model(X, y, classifier)
    # test_model(X_test, y_test, model)
    pkl.dump(model, open(output_model, "wb"))

if __name__ == '__main__':
    args = parse_args()
    image_dir = args.image_dir
    label_dir = args.label_dir
    classifier = args.classifier
    output_model = args.output_model
    main(image_dir, label_dir, classifier, output_model)
