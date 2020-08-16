# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 20:35:17 2016

@author: Mukundbj
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa

def cropImage(a=2,b=1.5):
    
    def opening(image):   
        #Using Opening Algorithm to remove noise
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
        save='F:/R Files/Biometric Sensor Project/Final Code/contour_opening/'
        return opening,save
    def erosion(image,num_iter=1):
        #Using erosion algorithm to remove noise and thin
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(image,kernel,iterations = num_iter)
        save='F:/R Files/Biometric Sensor Project/Final Code/contour_erosion/'
        return erosion,save
        
    os.chdir('F:/R Files/Biometric Sensor Project/Final Code/Images/')
    for file in os.listdir(os.curdir):
        if file[0]!='0':  #I have an issue with 'Thumbs.db'
             continue
        img = cv2.imread(file,0)
        img = cv2.medianBlur(img,5)
        th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                    cv2.THRESH_BINARY,13,2)
    
        # The code that follows draws a rectangle around the area of interest which, in our case, is
        # the hand. (x, y) is the location of the top-left corner of the rectangle, w is its width
        # and h is its height. I then crop only that portion of the image                    
        
        ret,thresh = cv2.threshold(img,90,255,0)
        im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        
        #Crop image using contour values        
        cropped_img=th2[y+20:y+h, x+20:x+w-20]
        cropped_img[cropped_img < 100] = 0
        cropped_img[cropped_img >= 100] = 255
        cropped_img=abs(255-cropped_img) #Inverts grayscale image 
        
        #img,to_save_base=opening(cropped_img)        
        img,to_save_base=erosion(cropped_img)
        
        #Plotting the image
        fig = plt.figure(frameon=False)
        fig.set_size_inches(a, b)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img,'gray', aspect = 'normal')
        to_save=to_save_base+file[:-4]+'.png'
        fig.savefig(to_save, dpi = 30)
        
def augment(path, to_save, batch_size = 20, rows = 480, cols = 800):
    os.chdir(path)
    arr = np.empty(shape=(batch_size, rows, cols), dtype=np.uint8)
    images_aug = np.empty(shape=(len(os.listdir(path)), rows, cols, 1), dtype=np.uint8)
    seq = iaa.Sequential([iaa.Fliplr(0.5)])
    i = 0
    j = 0
    for file in sort(os.listdir(path)):
        img = cv2.imread(file, 0)
        arr[j] = img
        i += 1
        j += 1
        if j == batch_size or i == len(os.listdir(path)):
            j = 0
            images_aug = np.vstack((images_aug, seq.augment_images(arr[:, :, :, np.newaxis])))
            arr = np.empty(shape=(batch_size, rows, cols), dtype=np.uint8)
    print arr.shape
    print images_aug.shape
    images_aug = images_aug[len(os.listdir(path)):, :, :, :]
    return images_aug
    #plot(images_aug, to_save)

def plot(aug, to_save_base, rows = 480, cols = 800, a = 5, b = 3):
    i = 1
    j = 0
    for img in aug:
        img = img.reshape(rows, cols)
        fig = plt.figure(frameon=False)
        fig.set_size_inches(a, b)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img,'gray', aspect = 'normal')
        to_save=to_save_base+str(i)+"_"+str(j)+'.jpg'
        fig.savefig(to_save, dpi = 160)
        j += 1
        if j == 10:
            i += 1
            j = 0

def sort(l):
    for i in range(len(l)):
        l[i] = int(l[i][:-6])
    l.sort()
    j = 0
    for i in range(len(l)):
        l[i] = str(l[i]) + "_" + str(j) + ".jpg"
        j += 1
        if j == 10:
            j = 0
    return l

def formPicFrame(aug):
    i=0
    f_image=open("H:/R Files/Biometric Sensor Project/Final Code/picdata_train.txt",'wb')
    f_image1=open("H:/R Files/Biometric Sensor Project/Final Code/picdata_val.txt",'wb')
    f_image2=open("H:/R Files/Biometric Sensor Project/Final Code/picdata_test.txt",'wb')
    f_image3=open("H:/R Files/Biometric Sensor Project/Final Code/picdata.txt",'wb')
    for img in aug:
        i+=1
        
        image=img.ravel()  #Transform 2D matrix into 1D array
        image.flags.writeable = True
        # Adjust values to be between 0 and 255 only
        print image.shape
        image[image < 100] = 0
        image[image >= 100] = 255
        f_image3.write(image.tostring(None))
        if i>7:
            if i==10:
                f_image2.write(image.tostring(None))
                i=0
                continue
            f_image1.write(image.tostring(None))
        else:
            f_image.write(image.tostring(None))

    f_image.close()
    f_image1.close()
    f_image2.close()
    f_image3.close()

augmented = augment("H:/R Files/Biometric Sensor Project/Final Code/Images/Test/","H:/R Files/Biometric Sensor Project/Final Code/Converted/")
formPicFrame(augmented)