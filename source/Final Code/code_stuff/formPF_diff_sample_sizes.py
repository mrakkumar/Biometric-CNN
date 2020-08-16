# -*- coding: utf-8 -*-
"""
Created on Sat Apr 08 12:47:13 2017

@author: HP
"""
import os
import numpy as np
from PIL import Image
from natsort import natsorted

def deleteContent(pfile):
    pfile.seek(0)
    pfile.truncate()
    
def write_to_file(lst, file):
    for x in lst:
        file.write(x.tostring(None))

def formPicFrame(path, text_path, train = 5, val = 3):
    f_image=open(text_path + "picdata_train.txt",'wb')
    f_image1=open(text_path + "picdata_val.txt",'wb')
    f_image2=open(text_path + "picdata_test.txt",'wb')
    log = open(text_path + "log.txt", 'a')
    
    deleteContent(log)
    
    os.chdir(path)
    total_n = 0
    sub_n = 0
    train_list = []
    test_list = []
    val_list = []
    
    for file in natsorted(os.listdir(os.curdir)):
        sub_n += 1
        if (sub_n == 11):
            sub_n = 1
            log.write("\n----------------------\n")
        img = Image.open(file)
        y = np.asarray(img.convert("L"))
        image=y.ravel()  #Transform 2D matrix into 1D array
        image.flags.writeable = True
        # Adjust values to be between 0 and 255 only
        #print image.shape
        image[image < 100] = 0
        image[image >= 100] = 255
        log.write(file + " read.\n")
        
        if sub_n <= train:
            train_list.append(image)
            log.write(file + " written to train.\n")
        elif sub_n <= train + val:
            val_list.append(image)
            log.write(file + " written to val.\n")
        else:
            test_list.append(image)
            log.write(file + " written to test.\n")
            
    write_to_file(train_list, f_image)
    write_to_file(val_list, f_image1)
    write_to_file(test_list, f_image2)

    f_image.close()
    f_image1.close()
    f_image2.close()
    
#img_path = "H:/R Files/Biometric Sensor Project/Final Code/new_augment_src/"
#text_path = "H:/R Files/Biometric Sensor Project/Final Code/"
#formPicFrame(img_path, text_path)