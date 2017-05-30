import os
import sys
import random

random.seed(1996)

import numpy as np
import pandas as pd

from scipy.stats import itemfreq 

from skimage import io
from skimage.transform import rotate
from skimage.feature import local_binary_pattern

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from tqdm import tqdm


import warnings
warnings.filterwarnings('ignore')

TRAIN_PATH = '../input/train/'
TEST_PATH  = '../input/test/'

LOAD_NB_MAX_TRAINING_DATA = 2295

train_labels = pd.read_csv('../input/train_labels.csv')


def load_img(n, training=True):
    return io.imread((TRAIN_PATH if training else TEST_PATH) + str(n) + '.jpg')

def extract_rgb(img): 
    return img[:,:,0], img[:,:,1], img[:,:,2]

def rgb2gray(rgb):
    r, g, b = extract_rgb(rgb)
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

LBP_RADIUS = 3
LBP_NB_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'

def get_lbp(img_gray_scale, radius=3):
    return local_binary_pattern(
        img_gray_scale,
        8 * radius,
        radius,
        LBP_METHOD
    )

def get_hist(lst):
    x = itemfreq(lst)
    hist = x[:, 1]/sum(x[:, 1])
    return hist

def get_lbp_hist(img, radius=3):
    return get_hist(get_lbp(rgb2gray(img), radius=radius).reshape(-1))

def load_lbp_features(img_number, training=True):
    img = load_img(img_number, training=training)
    descriptor = list(get_lbp_hist(img, 3)) #+ list(get_lbp_hist(img, 1)) + list(get_lbp_hist(img, 5)) + list(get_lbp_hist(img, 7))
    return descriptor

#load data
training_labels = np.array(list(train_labels.drop("name", axis=1)["invasive"]))[:LOAD_NB_MAX_TRAINING_DATA]

training_data = []
for i in tqdm(range(len(training_labels[:LOAD_NB_MAX_TRAINING_DATA]))):
    training_data.append(load_lbp_features(str(i + 1)))

training_data = np.array(training_data)

print("mean : ", training_data.mean(axis=0))
print("std  : ", training_data.std(axis=0))

#normalize data
training_data = (training_data - training_data.mean(axis=0)) / training_data.std(axis=0)


joblib.dump(training_data, 'X.pkl', compress=9)
print("training data saved")

#shuffling data
training_set = list(zip(training_labels, training_data))
#random.shuffle(training_set)

print("training set size : ", len(training_set))

train_set, test_set = train_test_split(training_set, test_size=.2)

Y_train, X_train = zip(*train_set)
Y_test,  X_test  = zip(*test_set)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test  = np.array(X_test)
Y_test  = np.array(Y_test)

print("nb training set : ", len(Y_train))
print("nb testing  set : ", len(Y_test))

clf = SVC(probability=True)
clf.fit(X_train, Y_train)

print("accuracy : ", sum(clf.predict(X_test) == Y_test) / float(len(Y_test)))


joblib.dump(clf, 'X.pkl', compress=9)

print("classifier saved...")


