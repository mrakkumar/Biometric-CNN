# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:56:39 2017

@author: Mukundbj
"""

#import func
import os
import string
import cv2
import time
import formPF
import func
from nn_an import main_alexnet
from nn_ln import main_lenet
from nn_vgg import main_vgg
from RF import main

#os.chdir('/media/akshay/My Passport/R Files/Biometric Sensor Project/Final Code/')
original_dir = "H:/R Files/Biometric Sensor Project/Final Code/P_Test/"
to_save_base = "H:/R Files/Biometric Sensor Project/Final Code/Test_alex/"
text_path = "H:/R Files/Biometric Sensor Project/Final Code/"
wts_path = "H:/R Files/Biometric Sensor Project/Final Code/"
letter_list = list(string.ascii_lowercase)
os.chdir(original_dir)
#start = time.time()
#
for file in os.listdir(original_dir):
#    try:
        augmented = [cv2.imread(file, 0)]
        augmented += func.augment(original_dir + file, to_save_base)
        for i in range(len(augmented)):
            aug = augmented[i].reshape(480, 800)
            cropped = func.cropImage(aug, to_save_base, 29, 1, 50, 2)
            to_save = to_save_base + file[:-4] + letter_list[i] + ".jpg"
            func.plot(cropped, to_save)
        print file + " processed.\n"
#    except:
#        print file + " unsuccesful.\n"
#        continue

formPF.formPicFrame(to_save_base, text_path)
#print str(time.time() - start) + " seconds"
#acc = main_lenet(text_path, 5)
#acc = main(text_path, wts_path, 5)

#with open(text_path + "accuracy.txt", 'w') as f:
#    f.write(', '.join(acc))