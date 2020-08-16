# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 12:49:59 2017

@author: HP
"""

from imgaug.imgaug import augmenters as iaa
import numpy as np
import os
import cv2
from natsort import natsorted
from matplotlib import pyplot as plt

def plot(img,to_save,a=4,b=3):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(a, b)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img,'gray', aspect = 'normal')
    fig.savefig(to_save, dpi = 30)
    
def load_batches(img_path, batch_size):
    n = 0
    arr = []
    image_list = natsorted(os.listdir(img_path))
    for i in range(len(image_list)):
        n += 1
        if n > batch_size:
            print ("\n------- End of batch -----------\n")
            yield arr
            arr = []
            n = 1
        img = cv2.imread(img_path + image_list[i], 0)
        arr.append(img)
        print ("Image: " + image_list[i])
    yield arr
    
def augment(img_batch):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([iaa.Fliplr(0.5),
                          sometimes(iaa.Affine(
                                  rotate=(-15, 15),
                                  shear=(-10, 10),
                                  translate_px={"x": (-16, 16), "y": (-16, 16)},
                                  order=[0, 1]))
                        ])
#    seq1=iaa.Sequential([iaa.GaussianBlur(sigma=(0.7, 1.7)),
#                          iaa.AdditiveGaussianNoise(loc=0, scale=(0.07, 0.15)),
#                          iaa.ContrastNormalization((1, 2.0)),
#                          iaa.Affine(rotate=(-15, 15))])
#    seq2=iaa.Sequential([iaa.GaussianBlur(sigma=(0.7, 1.7)),
#                          iaa.AdditiveGaussianNoise(loc=0, scale=(0.07, 0.15)),
#                          iaa.ContrastNormalization((1, 2.0)),
#                          iaa.Affine(translate_px={"x": (-16, 16), "y": (-16, 16)})])
    images_aug = seq.augment_images(img_batch[:, :, :, np.newaxis])
    return images_aug
#    images_aug1 = seq1.augment_images(arr[np.newaxis, :, :, np.newaxis])
#    images_aug2 = seq2.augment_images(arr[np.newaxis, :, :, np.newaxis])
#    l = [images_aug, images_aug1, images_aug2]
#    for i in range(len(l)):
#        plot(l[i], to_save + "aug_" + str(i + 1) + ".jpg")
        
def process_images(img_path, to_save, batch_size = 10):
    img_batches = load_batches(img_path, batch_size)
    save_fnames = natsorted(os.listdir(img_path))
    res = []
    for batch in img_batches:
        batch = np.asarray(batch)
        augmented = augment(batch)
        for img in augmented:
            img = img[:, :, 0]
            res.append(img)
    for i in range(len(res)):
        plot(res[i], to_save + save_fnames[i])
    
img_path = "/media/akshay/My Passport/R Files/Biometric Sensor Project/Final Code/new_augment_src/"
save_path = "/media/akshay/My Passport/R Files/Biometric Sensor Project/Final Code/augment_test_results/"
process_images(img_path, save_path)