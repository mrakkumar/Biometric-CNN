#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run.py
    central control to different modules of the biometric-CNN project

Created on Wed Aug 19 00:48:03 2020

__author__      = nnarenraju
__copyright__   = Copyright 2020, biometric-CNN
__credits__     = 
__license__     = 
__version__     = 1.0.0
__maintainer__  = 
__email__       = 
__status__      = inProgress, inUsage


Github Repository: https://github.com/mrakkumar/Biometric-CNN

Documentation: NULL

"""

import os
import numpy
import logging
import pathlib
import pickle as pkl
import datetime as dt

## local
#import preprocess as prep
from model_parser import ModelParser

## TF
import tensorflow as tf
from keras.models import load_model


class Run():
    
    """
    params:
        
        # MODEL PARAMS
        special     - boolean   - enables/disables super_model usage
        Model       - object    - holds a model_parser object
        modelfile   - string    - points to model object .pkl file
        modelconfig - string    - points to modelconfig .txt file
        seed        - int       - Seed value for operation
        savemod     - Boolean   - Save the (un)compiled model in HDF5 format
        
        # DATA PARAMS
        datadir     - string    - PATH to dataset folder (w class-subdir)
        dataset     - object    - tf.data.Dataset object
        imgcount    - int       - number of images present in dataset
        batchsize   - int       - size of batch for performace method
        filetype    - string    - File extension of img files (def: jpg)
        imgshape    - tuple     - (height, width, 1) from input image
        nclasses    - int       - number of classes present in dataset
        height      - int       - height in pixels of input image
        width       - int       - width in pixels of input image
        
        # PRE-PROCESSING (maintains folder hierarchy)
        prepdir     - string    - path to pre-processed dataset
        augment     - boolean   - augments the pre-processed dataset
        
        # SPLIT PARAMS
        train       - objects   - Image objects (as defined by TF) of training set
        vals        - objects   - Image objects (as defined by TF) of validation set
        valsize     - float     - Proportion of validation set (min:0.0, max:0.99)
        
    public methods:
        
        make_dataset  - creates and shuffles the dataset as tf.data.Dataset object
        split         - splits dataset into train and vals based on valsize
        make_pair     - make (img, label) pairs using dataset
        performance   - configure performace (tf methods)
        get_model     - make and get an uncompiled model

    Central control to different modules of the biometric-CNN project.
    Methods which are on beta have not been included in this __doc__

    """

    def __init__(self, datadir, modelfile=None, modelconfig=None):
        
        ### Logging params ###
        self.log = logging.getLogger(__name__)        
        
        # special (enables super_model)
        self.special = False
        
        # Model params
        self.Model = None
        self.modelfile = modelfile
        self.modelconfig = modelconfig
        self.seed = None
        self.savemod = False
        
        # Data params
        self.datadir = pathlib.Path(datadir)
        self.dataset = None
        self.imgcount = 0
        self.batchsize = 32
        self.filetype = "jpg"
        self.imgshape = None
        self.nclasses = None
        # get dim automatically
        # currently manual
        self.height = None
        self.width = None

        # Pre-processing params
        self.prepdir = None
        self.augment = False
        
        # Split params
        self.train = None
        self.vals = None
        self.valsize = 0.2
        self.savefeat = False
        
        # experimental
        self.imread = None
        self.imgfile = None
        
        ### TF Params ###
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    def read(self):
        """ under experimentation """
        self.log.debug('method "read" currently under debugging status has been accessed.')
        return getattr(im, self.imread)(self.imgfile)

    def img_count(self):
        self.imgcount = len(list(self.datadir.glob('*/*.{}'.format(self.filetype))))

    def make_dataset(self):
        self.img_count()
        list_ds = tf.data.Dataset.list_files(str(self.datadir/'*/*'), shuffle=False)
        self.dataset = list_ds.shuffle(self.imgcount, reshuffle_each_iteration=False)
        self.log.info("Dataset has been created and shuffled")
        
    def get_classes(self):
        return numpy.array(sorted([item.name for item in self.datadir.glob('*') if item.name != "LICENSE.txt"]))

    def split(self):
        val_size = int(self.imgcount * self.valsize)
        self.train = self.dataset.skip(val_size)
        self.vals = self.dataset.take(val_size)
        self.log.info("Dataset split into training and validation sets")

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
        return tf.image.resize(img, [self.height, self.width])

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
        self.log.info("Training and validation data converted to (img, label) pairs")

    def get_pairs(self, dataset):
        # Use if fitting with (data, label) input
        # Returns: image_batch, label_batch
        return next(iter(dataset))

    ### Configure dataset for performance ###

    def configure_for_performance(self, ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(self.batchsize)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds

    def performance(self):
        self.train = self.configure_for_performance(self.train)
        self.vals = self.configure_for_performance(self.vals)
        self.log.info("Configure for performance tf method have been called")

    ### Pre-processing the dataset ###

    def preprocess(self):
        prepdir = prep.preprocess_images(self.datadir)
        # Sanity check
        if os.path.exists(self.prepdir):
            raise ValueError("Prep folder not created.")
        self.prepdir = pathlib.Path(prepdir)
        self.log.info("Pre-processing completed and path received")
    
    ########################### MODEL MANIPULATION ################################
    # This section contains function definitions pertaining to model reading from 
    # the model config file, saving & loading a compiled/uncompiled model and 
    # saving the required weights of a model.
    # All details pertaining to reading and understanding the model config file
    # is present in the model_parser module.
    ###############################################################################

    def get_model(self):
        # Get uncompiled model object from ModelParser
        self.nclasses = len(self.get_classes())
        self.imgshape = (self.height, self.width, 1)
        
        # Sanity check
        self.is_exists(self.modelconfig)
        # Connect to model_parser
        mp = ModelParser(self.modelconfig, self.nclasses, self.imgshape, self.seed, \
                         special=self.special)
        # Reads, cleans and loads the model config file
        mp.read_input()
        # Make an uncompiled model
        mp.make_model()
        self.Model = mp
        self.log.info("Model object has been successfully created")

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
        self.log.warning("Model has been deleted from object after saving. Load for reuse.")
        # Save ModelClass object
        self.write_pkl(self.Model, filename='saved_models/model_{}.pkl'.format(str(dt.datetime.now().time())))
        self.log.info("Model has been saved under saved_models/")

    def load(self):
        # Loading a compiled/raw model from HDF5
        # modelfile is a .pkl file with info about model stored
        self.is_exists(self.modelfile)
        mod = self.read_pkl(self.modelfile)
        # modelpath points to the unique HDF5 file
        self.is_exists(mod.modelpath)
        mod.model = load_model(mod.modelpath)
        self.Model = mod
        self.log.info("Model successfully loaded from modelfile")




