# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 11:45:22 2017

@author: Mukundbj
"""

import cv2
import numpy as np
import matplotlib
# Use the 'Agg' backend while running the code to make sure separate plot windows don't open
# for each image. Use the 'TkAgg' backend when debugging to inspect the images
matplotlib.use('Agg')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from imgaug import augmenters as iaa
from errno import EEXIST

# TODO:
#  1. Encapsulate functions inside a class definition
#  2. Normalize images

# Function to check if directory exists. If not, create it.
def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    try:
        os.makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and os.path.isdir(mypath):
            pass
        else:
            raise

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

def cropImage(image, a=13, b=2, crop=20, erode=1):
    img = cv2.medianBlur(image,5)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,a,b)

    # The code that follows draws a rectangle around the area of interest which, in our case, is
    # the hand. (x, y) is the location of the top-left corner of the rectangle, w is its width
    # and h is its height. I then crop only that portion of the image                    s
    ret,thresh = cv2.threshold(img,90,255,0)
    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
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

    # Not sure of the parameters we had used for erosion and dilation. The ones here only seem
    # to make it worse XP
    # cropped_img=erosion(cropped_img,3,erode)
    # cropped_img=dilation(cropped_img,6,erode)
    # cropped_img=erosion(cropped_img,4,3)
    return cropped_img

def plot_augment(aug, to_save_base, orig_img_path, output_shape, a = 5, b = 3):
    j = 0
    orig_img = orig_img_path[:-4]
    for img in aug:
        # Added a resize command here because I couldn't find it anywhere else - Akshay
        # img = cv2.resize(img, output_shape, interpolation=cv2.INTER_LINEAR)
        fig = plt.figure(frameon=False)
        fig.set_size_inches(a, b)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img, 'gray', aspect = 'auto')
        mkdir_p(to_save_base)
        to_save = os.path.join(to_save_base, '{}_{}.jpg'.format(orig_img, str(j)))
        fig.savefig(to_save, dpi = 160)
        plt.close()
        j += 1
        if j == 4:
            j = 0
    
def augment(arr):
    seq1 = iaa.Sequential([iaa.GaussianBlur(sigma=(0.7, 1.7)),
                          iaa.AdditiveGaussianNoise(loc=0, scale=(0.07, 0.15)),
                          iaa.ContrastNormalization((1, 2.0)),
                          iaa.Affine(shear=(-10, 10))])
    seq2 = iaa.Sequential([iaa.GaussianBlur(sigma=(0.7, 1.7)),
                          iaa.AdditiveGaussianNoise(loc=0, scale=(0.07, 0.15)),
                          iaa.ContrastNormalization((1, 2.0)),
                          iaa.Affine(rotate=(-15, 15))])
    seq3 = iaa.Sequential([iaa.GaussianBlur(sigma=(0.7, 1.7)),
                          iaa.AdditiveGaussianNoise(loc=0, scale=(0.07, 0.15)),
                          iaa.ContrastNormalization((1, 2.0)),
                          iaa.Affine(translate_px={"x": (-16, 16), "y": (-16, 16)})])
    images_aug1 = seq1.augment_images(arr)
    images_aug2 = seq2.augment_images(arr)
    images_aug3 = seq3.augment_images(arr)
    return [images_aug1, images_aug2, images_aug3]

# Added new function here to save images after augmentation
# Not sure if we augmented before or after cropping the images
def preprocess_images(img_dir, output_dir, output_shape = (480, 800)):
    for img_path in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, img_path), 0)
        label = img_path.split('_')[0]

        # First, crop the Region of Interest
        # Not sure where we 'resized' the cropped image originally, can only find a 'reshape' command
        # in the 'plot_augment' function
        cropped_img = cropImage(img, crop = 70)

        # Second, augment the images
        aug_images = [cropped_img]
        aug_images.extend(augment(cropped_img))

        # Third, save the cropped, augmented images
        plot_augment(aug_images, os.path.join(output_dir, label), img_path, output_shape)

# This main function is simply to test out on a small subset of our data. Disregard this
# when we are writing up the training code
if __name__ == '__main__':
    img_dir = '/home/akshay/PycharmProjects/Biometric-CNN/source/data_subset'
    output_dir = '/home/akshay/PycharmProjects/Biometric-CNN/source/processed_data_subset'
    preprocess_images(img_dir, output_dir)