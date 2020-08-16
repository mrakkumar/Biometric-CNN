# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 11:45:22 2017

@author: Mukundbj
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from imgaug import augmenters as iaa

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

def cropImage(image, to_save, a=13,b=2,crop=20,erode=1):
         
    img = cv2.medianBlur(image,5)
    plot(img, to_save + "med_blur.jpg")
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,a,b)

    # The code that follows draws a rectangle around the area of interest which, in our case, is
    # the hand. (x, y) is the location of the top-left corner of the rectangle, w is its width
    # and h is its height. I then crop only that portion of the image                    
    
    ret,thresh = cv2.threshold(th2,90,255,0)
    plot(th2, to_save + "threshold.jpg")
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt=None
    for x in contours:
        area = cv2.contourArea(x)
        if (area<15000):
            continue
        cnt = x
        break
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(th2,(x,y),(x+w,y+h),(0,255,0),5)
    plot(th2, to_save + "contour.jpg")
    #Crop image using contour values        
    cropped_img=th2[y+crop:y+h-crop, x+crop:x+w-crop]
    cropped_img[cropped_img < 100] = 0
    cropped_img[cropped_img >= 100] = 255
    cropped_img=abs(255-cropped_img) #Inverts grayscale image
    plot(cropped_img, to_save + "cropped.jpg")
    img=erosion(cropped_img,3,erode)
    plot(img, to_save + "erosion.jpg")
    img=dilation(img,6,erode)
    plot(img, to_save + "dilation.jpg")
    img=erosion(img,4,3)
    plot(img, to_save + "erosion_2.jpg")
    #img=np.rot90(img,3)
    return img

def plot(img,to_save,a=4,b=3):
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
    
def augment(path, to_save, rows = 480, cols = 800):
    #st = lambda aug: iaa.Sometimes(0.5, aug)
    arr = cv2.imread(path, 0)
    seq = iaa.Sequential([iaa.GaussianBlur(sigma=(0.7, 1.7)),
                          iaa.AdditiveGaussianNoise(loc=0, scale=(0.07, 0.15)),
                          iaa.ContrastNormalization((1, 2.0)),
                          iaa.Affine(shear=(-10, 10))])
    seq1=iaa.Sequential([iaa.GaussianBlur(sigma=(0.7, 1.7)),
                          iaa.AdditiveGaussianNoise(loc=0, scale=(0.07, 0.15)),
                          iaa.ContrastNormalization((1, 2.0)),
                          iaa.Affine(rotate=(-15, 15))])
    seq2=iaa.Sequential([iaa.GaussianBlur(sigma=(0.7, 1.7)),
                          iaa.AdditiveGaussianNoise(loc=0, scale=(0.07, 0.15)),
                          iaa.ContrastNormalization((1, 2.0)),
                          iaa.Affine(translate_px={"x": (-16, 16), "y": (-16, 16)})])
    images_aug = seq.augment_images(arr[np.newaxis, :, :, np.newaxis])
    images_aug1 = seq1.augment_images(arr[np.newaxis, :, :, np.newaxis])
    images_aug2 = seq2.augment_images(arr[np.newaxis, :, :, np.newaxis])
    l = [images_aug, images_aug1, images_aug2]
#    for i in range(len(l)):
#        plot(l[i], to_save + "aug_" + str(i + 1) + ".jpg")
    return l
    
#def sort(l, aug = 0):
#    let_list = []
#    for i in range(len(l)):
#        let_list += l[i][-6:-4]
#        l[i] = l[i][:-6]
#        l[]
#    l.sort()
#    j = 0
#    for i in range(len(l)):
#        if aug == 1:
#            j = str(j) + let_list[i]
#        l[i] = str(l[i]) + "_" + str(j) + ".jpg"
#        j += 1
#        if j == 10:
#            j = 0
#    print l
#    return l

def formPicFrame(path, text_path, train = 5, val = 3, test = 2):
    f_image=open(text_path + "picdata_train.txt",'wb')
    f_image1=open(text_path + "picdata_val.txt",'wb')
    f_image2=open(text_path + "picdata_test.txt",'wb')
    f_image3=open(text_path + "picdata.txt",'wb')
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
        f_image3.write(image.tostring(None))
        l = file[-5]

        if l == 'a':
            a_list.append(image)
        elif l == 'b':
            b_list.append(image)
        elif l == 'c':
            c_list.append(image)
        elif l == 'd':
            d_list.append(image)
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
        else:
            f_image2.write(a_list[i].tostring(None))
            f_image2.write(b_list[i].tostring(None))
            f_image2.write(c_list[i].tostring(None))
            f_image2.write(d_list[i].tostring(None))
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
    f_image3.close()