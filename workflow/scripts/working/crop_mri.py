#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 21:23:40 2024

@author: greydon
"""

import cv2
import numpy as np
from cairosvg import svg2png
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
import matplotlib
import glob
import os



#%%

debug = True

if debug:
	class dotdict(dict):
		"""dot.notation access to dictionary attributes"""
		__getattr__ = dict.get
		__setattr__ = dict.__setitem__
		__delattr__ = dict.__delitem__

	class Namespace:
		def __init__(self, **kwargs):
			self.__dict__.update(kwargs)

	isub = 'sub-P025'
	#data_dir = r'/media/greydon/lhsc_data/SEEG_rerun/derivatives'
	data_dir = r'/home/greydon/Documents/data/SEEG_peds/derivatives'

	input = dotdict({
			'img_dir': f'{data_dir}/trajGuide/derivatives/{isub}/summaries/*.svg',
	 })
	
	snakemake = Namespace(input=input)

out_dir=os.path.join(os.path.dirname(snakemake.input.img_dir),'cropped')
if not os.path.exists(out_dir):
	os.makedirs(out_dir)


for ifile in [x for x in glob.glob(snakemake.input.img_dir) if '_red' not in os.path.basename(x).lower()]:
	
	out_fname=os.path.join(out_dir,os.path.splitext(os.path.basename(ifile))[0]+".png")
	
	svg2png(open(ifile, 'rb').read(), write_to=open(out_fname, 'wb'))
	img_orig = cv2.imread(out_fname)
	
	img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
	
	# threshold 
	threshold = 0.05 * np.max(img)
	
	thresh = img.copy()
	thresh[img < threshold] = 0
	
	contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]
	big_contour = max(contours, key=cv2.contourArea)
	mask = np.zeros_like(thresh, dtype=np.uint8)
	cv2.drawContours(mask, [big_contour], 0, 255, -1)
	x,y,w,h = cv2.boundingRect(big_contour)
	
	img_crop=img_orig.copy()
	img_crop[mask==0]=0
	img_crop = img_crop[y:y+h, x:x+w]
	
	cv2.imwrite(out_fname, img_crop)


