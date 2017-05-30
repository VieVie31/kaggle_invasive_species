import random

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

import warnings
warnings.filterwarnings('ignore')

#reproductibility
random.seed(1996)
np.random.seed(1996)

#load training labels
train_labels = pd.read_csv('../input/train_labels.csv')
training_labels = np.array(list(train_labels.drop("name", axis=1)["invasive"]))

#load training data (allready normalized)
training_data = joblib.load("invasive_species_lbp_training_data.pkl")

print("training set size : ", len(training_data))

#shuffling data
training_set = list(zip(training_labels, training_data))
random.shuffle(training_set)


#split training set
train_set, test_set = train_test_split(training_set, test_size=.1)

Y_train, X_train = zip(*train_set)
Y_test,  X_test  = zip(*test_set)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test  = np.array(X_test)
Y_test  = np.array(Y_test)

print("nb training set : ", len(Y_train))
print("nb testing  set : ", len(Y_test))


#train a SVC classifier
clf_svc = SVC(probability=True)
clf_svc.fit(X_train, Y_train)

print("SVC                    accuracy : ", sum(clf_svc.predict(X_test) == Y_test) / float(len(Y_test)))


#train a RandomForestClassifier classifier
clf_rfc = RandomForestClassifier()
clf_rfc.fit(X_train, Y_train)

print("RandomForestClassifier accuracy : ", sum(clf_rfc.predict(X_test) == Y_test) / float(len(Y_test)))

#load testing data (allready normalized)
testing_data = joblib.load("invasive_species_lbp_testing_data.pkl")
testing_predicted_labels_proba = clf_svc.predict_proba(testing_data)[:,1]

#save the ouput for kaggle in csv
s = "name,invasive\n"
for i, v in enumerate(testing_predicted_labels_proba):
    s += str(i + 1) + ',' + str(v) + chr(10)

f = open('submit.csv', 'w')
f.write(s)
f.close()

print("done !")


