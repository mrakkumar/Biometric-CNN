#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 10:37:56 2017

@author: akshay
"""

from skimage import feature
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import LinearSVC

def plot(img,to_save,a=4,b=3):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(a, b)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, aspect = 'auto')
    fig.savefig(to_save, dpi = 30)

def describe(image, P, R, method, a, b):
    im2, contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt=None
    for x in contours:
        area = cv2.contourArea(x)
        if (area<15000):
            continue
        cnt = x
        break
    
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),5)
    plt.imshow(image)
#    lbp = feature.local_binary_pattern(image, P, R, method=method).astype(np.uint8)
#    th2 = cv2.adaptiveThreshold(lbp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,a,b)
#    ret, thresh = cv2.threshold(th2, 127, 255, cv2.THRESH_BINARY)
    return image

def form_hist(lbp, numPoints, eps = 1e-7):
    (hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, numPoints + 3),
			range=(0, numPoints + 2))
 
	# normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)

	# return the histogram of Local Binary Patterns
    return hist
  
def get_data(path, P, R):
    data = []
    labels = []
    for file in os.listdir(path):
        if file == '.png':
            continue
        img = cv2.imread(path + file, 0)
        desc = describe(img, P, R)
        hist = form_hist(desc, P)
        data.append(hist)
        print (file)
        labels.append(int(file[:-6]))
    return data, labels

def generate_lbp_images(path, save_path, P = 8, R = 8, method = 'ror', a = 29, b = 1):
    for file in os.listdir(path):
        try:
            img = cv2.imread(path + file, 0)
            final = describe(img, P, R, method, a, b)
            plot(final, save_path + file)
            print (file + " saved successfully.")
        except ValueError:
            print (file + " unsuccesful. NoneType image read.")
    
path = "/media/akshay/My Passport/R Files/Biometric Sensor Project/Final Code/Test/"
save_path = "/media/akshay/My Passport/R Files/Biometric Sensor Project/Final Code/Test_LBP/"
test_path = "/media/akshay/My Passport/R Files/Biometric Sensor Project/Final Code/Predicting_LBP/"
generate_lbp_images(path, save_path, a = 7, b = 2, method = 'uniform')