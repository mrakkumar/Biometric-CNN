# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 11:45:22 2017

@author: Mukundbj
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import os

def dilation(image,k,num_iter):   
    #Using Opening Algorithm to remove noise
    kernel = np.ones((k,k),np.uint8)
    dilation = cv2.dilate(image,kernel,iterations = num_iter)
    return dilation


def erosion(image,k,num_iter):
    #Using erosion algorithm to remove noise and thin
    kernel = np.ones((k,k),np.uint8)
    erosion = cv2.erode(image,kernel,iterations = num_iter)
    return erosion

def cropImage(path,a=13,b=2,crop=20,erode=1):
            
    img = cv2.imread(path,0)
    img = cv2.medianBlur(img,5)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,a,b)

    # The code that follows draws a rectangle around the area of interest which, in our case, is
    # the hand. (x, y) is the location of the top-left corner of the rectangle, w is its width
    # and h is its height. I then crop only that portion of the image                    
    
    ret,thresh = cv2.threshold(img,90,255,0)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt=None
    for x in contours:
        area = cv2.contourArea(x)
        if (area<15000):
            continue
        cnt = x
        break
    x,y,w,h = cv2.boundingRect(cnt)
    
    #Crop image using contour values        
    cropped_img=th2[y+crop:y+h-crop, x+crop:x+w-crop]
    cropped_img[cropped_img < 100] = 0
    cropped_img[cropped_img >= 100] = 255
    cropped_img=abs(255-cropped_img) #Inverts grayscale image
    img=erosion(cropped_img,3,erode)
    img=dilation(img,4,erode)
    img=erosion(img,4,3)
    #img=np.rot90(img,3)
    return img

def plot(img,to_save,a=2,b=1.5):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(a, b)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img,'gray', aspect = 'normal')
    fig.savefig(to_save, dpi = 30)
    
def plot_augment(aug, to_save_base, rows = 480, cols = 800, a = 5, b = 3):
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
    
def augment(path, to_save, batch_size = 20, rows = 480, cols = 800):
    os.chdir(path)
    arr = np.empty(shape=(batch_size, rows, cols), dtype=np.uint8)
    images_aug = np.empty(shape=(len(os.listdir(path)), rows, cols, 1), dtype=np.uint8)
    st = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([st(iaa.GaussianBlur(sigma=(0.7, 1.7))),
                          st(iaa.AdditiveGaussianNoise(loc=0, scale=(0.07, 0.15))),
                          st(iaa.ContrastNormalization((0.5, 2.0))),
                          st(iaa.Affine(shear=(-10, 10)))])
    i = 0
    j = 0
    for file in sort(os.listdir(path)):
        img = cv2.imread(file, 0)
        arr[j] = img
        i += 1
        j += 1
        if j == batch_size or i == len(os.listdir(path)):
            images_aug = np.vstack((images_aug, seq.augment_images(arr[:j, :, :, np.newaxis])))
            arr = np.empty(shape=(batch_size, rows, cols), dtype=np.uint8)
            j = 0
    print arr.shape
    print images_aug.shape
    images_aug = images_aug[len(os.listdir(path)):, :, :, :]
    plot_augment(images_aug, to_save)

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

def formPicFrame(path):
    i=0
    f_image=open("H:/R Files/Biometric Sensor Project/Final Code/picdata_train.txt",'wb')
    f_image1=open("H:/R Files/Biometric Sensor Project/Final Code/picdata_val.txt",'wb')
    f_image2=open("H:/R Files/Biometric Sensor Project/Final Code/picdata_test.txt",'wb')
    f_image3=open("H:/R Files/Biometric Sensor Project/Final Code/picdata.txt",'wb')
    os.chdir(path)
    for file in sort(os.listdir(os.curdir)):
        img = cv2.imread(file, 0)
        i+=1
        y = np.asarray(img)
        image=y.ravel()  #Transform 2D matrix into 1D array
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