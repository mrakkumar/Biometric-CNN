#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py
    End-user interface for the biometric-CNN project 

Created on Wed Aug 19 00:48:03 2020

__author__      = 
__copyright__   = Copyright 2020, biometric-CNN
__credits__     = 
__license__     = 
__version__     = 1.0.0
__maintainer__  = nnarenraju
__email__       = 
__status__      = inProgress


Github Repository: https://github.com/mrakkumar/Biometric-CNN

Documentation: NULL

"""

import os
import logging
import pathlib
import argparse
import numpy as np
import pickle as pkl
import datetime as dt

## local
import imgread as im
import preprocess as prep
from model_parser import ModelParser

## TF
import tensorflow as tf
from keras.models import load_model

class Run():
    
    """
    params:

        modelfile - string    - Points to model .txt file
        augment   - boolean   - Do or ignore augment method
        valsize   - float     - Proportion of validation set (min:0.0, max:0.99)
        datadir   - string    - PATH to dataset folder (w class-subdir)
        imgcount  - integer   - Count of total number of images in datadir
        imgfile   - string    - PATH to single image file
        filetype  - string    - File extension of img files (def: jpg)
        train     - objects   - Image objects (as defined by TF) of training set
        vals      - objects   - Image objects (as defined by TF) of validation set
        savefeat  - Boolean   - Save CNN features
        imread    - string    - Image read method (refer imgread.py)
        seed      - int       - Seed value for operation
        savemod   - Boolean   - Save the (un)compiled model in HDF5 format
        
    methods:

        read      -        - 

    Creates an object that can be input as a Structure to a c-types 
    function written to call a c-function.

    """

    def __init__(self, datadir, modelfile=None, modelconfig=None):

        # Model params
        self.Model = None
        self.modelfile = modelfile
        self.modelconfig = modelconfig
        self.seed = None
        self.savemod = False

        # Data params
        self.datadir = pathlib.Path(datadir)
        self.imgcount = 0
        self.batchsize = 32
        self.filetype = "jpg"
        self.imgshape = None
        self.nclasses = None

        # Pre-processing params
        self.prepdir = None
        self.augment = False
        
        # Fitting params
        self.train = None
        self.vals = None
        self.valsize = 0.2
        self.savefeat = False
        
        # Special
        self.imread = None
        self.imgfile = None
        
        ### TF Params ###
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        

    def read(self):
        return getattr(im, self.imread)(self.imgfile)

    def img_count(self):
        self.imgcount = len(list(self.datadir.glob('*/*.{n}'.format(self.filetype))))

    def make_dataset(self):
        list_ds = tf.data.Dataset.list_files(str(self.datadir/'*/*'), shuffle=False)
        return list_ds.shuffle(self.imgcount, reshuffle_each_iteration=False)

    def get_classes(self):
        return np.array(sorted([item.name for item in self.datadir.glob('*') if item.name != "LICENSE.txt"]))

    def split(self, list_ds):
        val_size = int(self.imgcount * self.valsize)
        self.train = list_ds.skip(val_size)
        self.vals = list_ds.take(val_size)  

    ### converts a file path to an (img, label) pair ###
    
    def get_label(self, file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == self.get_classes()
        # Integer encode the label
        return tf.argmax(one_hot)
      
    def decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # resize the image to the desired size
        img_height, img_width, _ = self.train.take(1)[0].shape
        return tf.image.resize(img, [img_height, img_width])

    def process_path(self, file_path):
        label = self.get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

    def make_pair(self):
        """ converts a file path to an (img, label) pair """
        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        self.train = self.train.map(self.process_path, num_parallel_calls=self.AUTOTUNE)
        self.vals = self.vals.map(self.process_path, num_parallel_calls=self.AUTOTUNE)

    def get_pairs(self, dataset):
        # Use if fitting with (data, label) input
        # Returns: image_batch, label_batch
        return next(iter(dataset))

    ### Configure dataset for performance ###

    def configure_for_performance(self, ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds

    def performance(self):
        self.train = self.configure_for_performance(self.train)
        self.vals = self.configure_for_performance(self.vals)

    ### Pre-processing the dataset ###

    def preprocess(self):
        self.prepdir = prep.preprocess_images(self.datadir)
        # Sanity check
        if os.path.exists(self.prepdir):
            raise ValueError("Prep folder not created.")

    
    ########################### MODEL MANIPULATION ################################
    # This section contains function definitions pertaining to model reading from 
    # the model config file, saving & loading a compiled/uncompiled model and 
    # saving the required weights of a model.
    # All details pertaining to reading and understanding the model config file
    # is present in the model_parser module.
    ###############################################################################

    def get_model(self):
        # Get uncompiled model object from ModelParser
        self.nclasses = len(self.get_classes)
        self.imgshape = self.train.take(1)[0].shape
        
        # Sanity check
        self.is_exists(self.modelconfig)
        # Connect to model_parser
        mp = ModelParser()
        mc.make_model(self.imgshape, self.nclasses, seed=self.seed)
        self.Model = mc

    def is_exists(self, filename):
        if not os.path.exists(filename):
            raise NameError("'{}' path does not exist.".format(filename))

    def write_pkl(self, vals, filename):
        # Write a pickle file with dict
        with open(filename, 'wb') as handle:
            pkl.dump(vals, handle)

    def read_pkl(self, file):
        # Read a pickle file and return dict
        with open(file, 'rb') as foo:
            data = pkl.load(foo)
        return data

    def save_model(self):
        # Sanity check
        if not os.path.isdir("saved_models"):
            os.mkdir("saved_models")
            os.mkdir("saved_models/models")
        self.Model.modelpath = 'saved_models/models/model_{}.h5'.format(str(dt.datetime.now().time()))
        self.Model.model.save(self.Model.modelpath)
        del self.Model.model #remove this if needed
        # Save ModelClass object
        self.write_pkl(self.Model, filename='saved_models/model_{}.pkl'.format(str(dt.datetime.now().time())))

    def load(self):
        # Sanity check
        self.is_exists(self.modelfile)
        mc = self.read_pkl(self.modelfile)
        self.is_exists(mc.modelpath)
        mc.model = load_model(mc.modelpath)
        self.Model = mc




