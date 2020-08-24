#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import argparse

from PIL import Image
from numpy import asarray
from matplotlib import image
from timeit import default_timer

def timer(func):
	def wrapper(filename):
		init = default_timer()
		func(filename)
		end = default_timer()
		print("Elapsed time: ", end-init)
	return wrapper

def cv2_read(filename):
	img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

def pillow(filename):
	img = Image.open(filename)
	img = asarray(img)

def matplotlib(filename):
	img = image.imread(filename)

def pillow-simd(filename):
	# 2-4 times faster than PIL
	# Might produce better results.
	pass


if __name__ == "__main__":
    
    p = argparse.ArgumentParser()
    
    p.add_argument("-i", type=str, nargs="+", default=False,
                   help="Input image file (.jpeg preferable)")
    p.add_argument("-timeit", action="store_true",
                   help="Time the required method")
    p.add_argument("--cv2", action="store_true",
                   help="Load image file using cv2.imread on GRAYSCALE")
    p.add_argument("--pillow", action="store_true",
                   help="Load image file using cv2.imread on GRAYSCALE")
    p.add_argument("--plt", action="store_true",
                   help="Load image file using cv2.imread on GRAYSCALE")
    p.add_argument("-o", type=str, default="log.dat",
                   help="Output from imgread")
    
    args = p.parse_args()
    
    # Sanity Check
    if not args.i:
        raise ValueError("No input data provided!")
    if not any([args.cv2, args.plt, args.pillow]):
    	print("userwarning: No load method specified. Defaulting to cv2.")
    	args.cv2 = True

    # Passing control
    if args.cv2:
    	if args.timeit:
    		cv2_read=timer(cv2_read)
    	cv2_read(args.i[0])
    elif args.pillow:
    	if args.timeit:
    		pillow=timer(pillow)
    	pillow(args.i[0])
    elif args.plt:
    	if args.timeit:
    		matplotlib=timer(matplotlib)
    	matplotlib(args.i[0])

    # EOF
    print("Done.")