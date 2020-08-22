#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model_parser.py
    model parser reads model config file and creates uncompiled model

Created on Sat Aug 21 17:12:22 2020

__author__      = nnarenraju
__copyright__   = Copyright 2020, biometric-CNN
__credits__     = nnarenraju
__license__     = Apache License 2.0
__version__     = 1.0.0
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = inProgress, Debugging


Github Repository: NULL

Documentation: NULL

"""

import re
import json
import inspect
from tensorflow.keras import layers, models, optimizers, initializers

class ModelParser():

    def __init__(self, modelconfig, nclasses, imgshape, seed=None):
        # Model object as returned after parsing
        self.model = None
        self.layers = None
        self.seed = seed
        # Dataset params
        self.nclasses = nclasses
        self.imgshape = imgshape
        # Config .txt file for model
        self.config = modelconfig
        # Path to model save location
        # Model not saved by default
        self.modelpath = None
        # Status of model (compiled/uncompiled)
        self.compiled = False
    
    def read_input(self):
        # Read input file and strip newline characters
        with open(self.config, 'r') as file:
            data = file.read()
            # Clean up the model file
            # Remove comments and carriage returns
            re.sub('^#.*?\n', '', data)
            data.replace('\n', '')
                
        # Loading the above data into a dict obj
        self.layers = json.loads(data)
        
        # Stroring the above dict as variables
        # Use this only if exec will not cause problems
        """
        for (n, v) in layers.items():
            exec('%s=%s' % (n, repr(v)))
        """
    
    def make_model(self):
        """
        Make a tf model using modelconfig
        """
        
        # Initializer and first_layer params
        # Initializer is not inspected as seed is always provided/no other params possible
        initializer = getattr(initializers, self.layers[0]['kernel_initializer'])(seed=self.seed)
        # Start model scope
        model = models.Sequential()
        
        # pop the optimizer if present
        optimizer = None
        for n, layer in enumerate(self.layers):
            if layer['layername'] in dir(optimizers):
                optimizer = getattr(optimizers, layer['layername'])
                opt_params = self.layers.pop(n)
                del opt_params['layername']
                break
        else:
            optimizer = None
        
        # Iterating through all layers
        for layer in self.layers:
            tf_layer = getattr(layers, layer['layername'])
            del layer['layername']
            
            # Sanity check
            allparams = inspect.getfullargspec(tf_layer).args
            if not set(layer.keys()) <= set(allparams):
                raise NameError("One or more params passed to layer does not exist.")
            # First layer exception
            if 'kernel_initializer' in layer.keys():
                layer['kernel_initializer'] = initializer
            
            # Pass all params to the tf layer function
            # as layer is a dict, params are automatically mapped by **kwargs
            model.add(tf_layer(layer))
        
        # Run the optimizer if requested
        if optimizer:
            optimizer(opt_params)
    
        return model
    
    
    
    
    
    
    
    
    
    
    