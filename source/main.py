#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
    main function: (run.py, model_parser.py, preprocess.py)

Created on Sat Aug 22 23:35:26 2020

__author__      = 
__copyright__   = Copyright 2020, biometric-CNN
__credits__     = 
__license__     = Apache License 2.0
__version__     = 1.0.0
__maintainer__  = 
__email__       = 
__status__      = inProgress, Debugging


Github Repository: NULL

Documentation: NULL

"""

from run import Run

if __name__ == "__main__":
    
    datadir = '/home/nnarenraju/.keras/datasets/flower_photos'
    modelconfig = 'model.test'
    r = Run(datadir=datadir, modelconfig=modelconfig)
    
    r.make_dataset()
    r.split()
    r.make_pair()
    r.performance()
    r.get_model()
    