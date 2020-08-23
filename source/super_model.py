#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
super_model.py
    Maximum priority model configuration. Overrides modelconfig and modefile.
    Return -1 to disable

Created on Mon Aug 24 00:01:04 2020

__author__      = nnarenraju
__copyright__   = Copyright 2020, biometric-CNN
__credits__     = nnarenraju
__license__     = 
__version__     = 1.0.0
__maintainer__  = 
__email__       = 
__status__      = inUsage


Github Repository: NULL

Documentation: NULL

"""

from tensorflow.keras import layers, models, initializers

def _special_(imgshape, nclasses, seed=None):
    
    """
    Special model file:
        0. Set "Run.special = True" to enable special model
        1. Overrides the presence of any modelconfig or loaded modelfile
        2. Manual description of the sequential model
        3. Do NOT include compilation
        4. Initializers have to be included
    
    """
    
    # Add your model here
    # A sample model has been added below for reference
    initializer = initializers.GlorotUniform(seed = seed)
    model = models.Sequential()
    model.add(layers.Conv2D(16, (5, 5), activation = 'relu', input_shape=imgshape, \
                            kernel_initializer = initializer))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation = 'relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Flatten())
    # Output
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(nclasses))
    model.add(layers.Dropout(0.5))
    model.add(layers.Softmax())
    
    return model
