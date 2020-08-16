# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 04:04:11 2017

@author: HP
"""

import numpy as np
from formPF_diff_sample_sizes import formPicFrame
from catboost import CatBoostClassifier
import os
import time
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

def load_dataset(path, n_out):

    def load_images(filename):
        with open(path + filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 2700)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        return data / np.float32(256)

    def load_labels(start=0,stop=50,repeat=1):
        data=np.arange(start,stop,1,np.int16)
        data=np.repeat(data,repeat)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    X_train = load_images('picdata_train.txt')
    y_train = load_labels(0,n_out,5)
    X_val=load_images('picdata_val.txt')
    y_val=load_labels(0,n_out,3)

    return X_train, y_train, X_val, y_val

def catboost(X_train, y_train, X_val, y_val):
    start = time.time()
    model = CatBoostClassifier(loss_function='MultiClass')
    model.fit(X_train, y_train)
    training_time = time.time() - start
    predictions = model.predict(X_val)
    pred_time = time.time() - training_time
    return predictions, round(accuracy_score(y_val, predictions), 2), training_time, pred_time

def xgboost(X_train, y_train, X_val, y_val):
    start = time.time()
    model = XGBClassifier()
    model.fit(X_train, y_train)
    t_time = time.time()
    training_time = t_time - start
    y_pred = model.predict(X_val)
    predictions = [round(value) for value in y_pred]
    pred_time = time.time() - t_time
    return predictions, round(accuracy_score(y_val, predictions), 2), training_time, pred_time

def SVM(X_train, y_train, X_val, y_val, rand=None):
    start = time.time()
    model = svm.LinearSVC(random_state = rand)
    model.fit(X_train, y_train)
    t_time = time.time()
    training_time = t_time - start
    predictions = model.predict(X_val)
    pred_time = time.time() - t_time
    return predictions, round(accuracy_score(y_val, predictions), 2), training_time, pred_time

def GridCV(X_train, y_train, X_val, y_val):
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
    start = time.time()
    grid.fit(X_train, y_train)
    t_time = time.time()
    training_time = t_time - start
    predictions = grid.predict(X_val)
    pred_time = time.time() - t_time
    return predictions, round(accuracy_score(y_val, predictions), 2), training_time, pred_time

img_path = "H:/R Files/Biometric Sensor Project/Final Code/new_augment_src/"
text_path = "H:/R Files/Biometric Sensor Project/Final Code/"
n_out = len(os.listdir(img_path)) / 10
#formPicFrame(img_path, text_path)
#print ("---------- Done forming pic frame ----------")
X_train, y_train, X_val, y_val = load_dataset(text_path, n_out)
pred, acc, tr_time, pr_time = SVM(X_train, y_train, X_val, y_val)
print ("Accuracy: " + str(acc * 100.0) + "%")
print ("Training time: " + str(tr_time) + " seconds.")
print ("Prediction time: " + str(pr_time) + " seconds.")