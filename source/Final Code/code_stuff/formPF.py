# -*- coding: utf-8 -*-
"""
Created on Sat Apr 08 12:47:13 2017

@author: HP
"""
import os
import numpy as np
import cv2

def formPicFrame(path, text_path, train = 5, val = 3):
    f_image=open(text_path + "picdata_train.txt",'wb')
    f_image1=open(text_path + "picdata_val.txt",'wb')
    f_image2=open(text_path + "picdata_test.txt",'wb')
#    f_image3=open(text_path + "picdata.txt",'wb')
    a_list = []
    b_list = []
    c_list = []
    d_list = []
    os.chdir(path)
    #print os.listdir(os.curdir)
    
    for file in os.listdir(os.curdir):
        img = cv2.imread(file, 0)
        y = np.asarray(img)
        image=y.ravel()  #Transform 2D matrix into 1D array
        image.flags.writeable = True
        # Adjust values to be between 0 and 255 only
        #print image.shape
        image[image < 100] = 0
        image[image >= 100] = 255
        l = file[-5]

        if l == 'a':
            a_list.append(image)
        elif l == 'b':
            b_list.append(image)
        elif l == 'c':
            c_list.append(image)
        elif l == 'd':
            d_list.append(image)
        
        print (file + " read.\n")
        
    n = 0
    for i in range(len(a_list)):
        if n < train:
            f_image.write(a_list[i].tostring(None))
            f_image.write(b_list[i].tostring(None))
            f_image.write(c_list[i].tostring(None))
            f_image.write(d_list[i].tostring(None))
        elif n < train + val:
            f_image1.write(a_list[i].tostring(None))
            f_image1.write(b_list[i].tostring(None))
            f_image1.write(c_list[i].tostring(None))
            f_image1.write(d_list[i].tostring(None))
        
        f_image2.write(a_list[i].tostring(None))
        n += 1
        if n == 10:
            n = 0
    
#        if i>7:
#            if i==10:
#                f_image2.write(image.tostring(None))
#                i=0
#                continue
#            f_image1.write(image.tostring(None))
#        else:
#            f_image.write(image.tostring(None))

    f_image.close()
    f_image1.close()
    f_image2.close()
#    f_image3.close()