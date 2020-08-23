#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from run import Run

"""
Please use the following to download flowers dataset:
    
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url, 
                                   fname='flower_photos', 
                                   untar=True)
"""

if __name__ == "__main__":
    
    # Logging sanity check
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    # Logging
    logging.basicConfig(filename='run.log', filemode='w', \
                        format='%(name)s@%(asctime)s - %(levelname)s - %(message)s',\
                        datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    logging.info("Start run module")
    
    # Locations
    datadir = '/home/nnarenraju/.keras/datasets/flower_photos'
    modelconfig = 'model.test'
    
    r = Run(datadir=datadir, modelconfig=modelconfig)
    r.special = True
    # set height and width
    r.height = 180
    r.width = 180
    
    r.make_dataset()
    r.split()
    r.make_pair()
    r.performance()
    r.get_model()
    r.save_model()
    