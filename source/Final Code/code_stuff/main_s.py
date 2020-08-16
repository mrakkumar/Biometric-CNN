# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:56:39 2017

@author: Mukundbj
"""

#import func
#import os
import string
#import func
#import cv2
#import time
#import formPF
from RF import main
from nn_vgg import main_vgg

#os.chdir('/media/akshay/My Passport/R Files/Biometric Sensor Project/Final Code/')
original_dir = "/home/akkumar/Images/test/"
to_save_base = "/home/akkumar/Images/Converted/"
text_path = "/home/akkumar/"
wts_path = "/home/akkumar/imgaug/"
letter_list = list(string.ascii_lowercase)
#os.chdir(original_dir)
#start = time.time()
#
#for file in os.listdir(original_dir):
#    try:
#        augmented = [cv2.imread(file, 0)]
#        augmented += func.augment(original_dir + file)
#        for i in range(len(augmented)):
#            aug = augmented[i].reshape(480, 800)
#            cropped = func.cropImage(aug, 29, 1, 50, 2)
#            to_save = to_save_base + file[:-4] + letter_list[i] + ".jpg"
#            func.plot(cropped, to_save)
#        print file + " processed.\n"
#    except:
#        print file + " unsuccesful.\n"
#        continue
#
#formPF.formPicFrame(to_save_base, text_path)
#print str(time.time() - start) + " seconds"
#acc = main(text_path, wts_path, 403)
acc = main(text_path, wts_path, 403)
print acc

with open(text_path + "accuracy.txt", 'w') as f:
    f.write(', '.join(acc))