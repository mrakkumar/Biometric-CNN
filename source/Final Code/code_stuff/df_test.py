# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 09:53:17 2017

@author: Akshay
"""

import os
import numpy as np
import cv2
from func import sort

def create_df(path):
    os.chdir(path)
    f_list = sort(os.listdir(path))
    length = len(f_list)
    test_file = cv2.imread(f_list[0], cv2.IMREAD_GRAYSCALE)
    arr = np.zeros((length, test_file.shape[0], test_file.shape[1], 3))
    i = 0
    for fname in f_list:
        img = cv2.imread(fname)
        arr[i] = img
        i += 1
    return arr
        
path = "H:/R Files/Biometric Sensor Project/Final Code/df_test/"
to_save_base = "H:/R Files/Biometric Sensor Project/Final Code/df_test/"
df = create_df(path)
df.tofile(to_save_base + "df_test.csv", sep = ",")