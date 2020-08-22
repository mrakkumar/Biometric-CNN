#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
foo.py
    Short abstract of foo.py
    Additional description

Created on Sat Aug 21 17:12:22 2020

__author__      = nnarenraju
__copyright__   = Copyright 2020, Project Name
__credits__     = nnarenraju
__license__     = Apache License 2.0
__version__     = 1.0.0
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = ['inProgress', 'inUsage', 'Archived', 'Debugging']


Github Repository: NULL

Documentation: NULL

"""

import re
import json

class ModelParser():

    def __init__(self, modelconfig):
        # Model object as returned after parsing
        self.model = None
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
        layers = json.loads(data)
        
        # Stroring the above dict as variables
        # Use this only if exec will not cause problems
        """
        for (n, v) in layers.items():
            exec('%s=%s' % (n, repr(v)))
        """
        
        # Returns all layers as dict object
        return layers
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    