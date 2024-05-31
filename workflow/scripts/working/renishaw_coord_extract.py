#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 13:24:58 2023

@author: greydon
"""

from PIL import Image, ImageEnhance, ImageOps
import easyocr
import os
import numpy as np
import pdfplumber
import numpy as np
import regex as re
import glob
import os
import datetime
import pandas as pd

def read_coords(image):
	reader = easyocr.Reader(['en'])
	results = reader.readtext(image)
	return results

def read_text_and_save_crops(image_path, additional_keywords=None, expansion_factor=6):
	
	output_folder=os.path.join(os.path.dirname(image_path),'content')
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	
	default_keywords = ['trajectory','target','trajectory length']
	keywords = default_keywords + [k.lower() for k in additional_keywords] if additional_keywords else default_keywords
	
	reader = easyocr.Reader(['en'])
	results = reader.readtext(image_path, detail=1)
	
	img = Image.open(image_path)
	
	keyword_bboxes = {k: None for k in keywords}
	
	for (bbox, text, prob) in results:
		text = text.lower()
		for keyword in keywords:
			if keyword in text:
				expansion_x = (bbox[2][0] - bbox[0][0]) * expansion_factor
				expansion_y = (bbox[2][1] - bbox[0][1]) * expansion_factor
				expanded_bbox = [
					max(0, bbox[0][0] - expansion_x/8),  # Left
					max(0, bbox[0][1] - expansion_y/8),  # Top
					min(img.size[0], bbox[2][0] + expansion_x/1.5),  # Right
					min(img.size[1], bbox[2][1] + expansion_y)   # Bottom
				]

				if keyword_bboxes[keyword]:
					keyword_bboxes[keyword] = [
						min(keyword_bboxes[keyword][0], expanded_bbox[0]),  # Left
						min(keyword_bboxes[keyword][1], expanded_bbox[1]),  # Top
						max(keyword_bboxes[keyword][2], expanded_bbox[2]),  # Right
						max(keyword_bboxes[keyword][3], expanded_bbox[3])   # Bottom
					]
				else:
					keyword_bboxes[keyword] = expanded_bbox
	
	def save_crop(img, bbox, file_name):
		cropped_img = img.crop(bbox)  # Crop the image to the bounding box
		cropped_img.save(file_name)   # Save the cropped image as a PNG file
	
	saved_crops = {}
	for keyword, bbox in keyword_bboxes.items():
		if bbox:
			file_name = f"{keyword.replace(' ', '_')}.png"
			crop_path = os.path.join(output_folder, file_name)
			save_crop(img, bbox, crop_path)
			saved_crops[keyword.title()] = crop_path

	ocr_results = {}
	for keyword in ['target']:
		bbox=keyword_bboxes[keyword]
		if bbox:
			file_name = f"{keyword.replace(' ', '_')}.png"
			crop_path = os.path.join(output_folder, file_name)

			# Load the image from file
			img = Image.open(crop_path)

			# Enhance the contrast of the image
			enhancer = ImageEnhance.Contrast(img)
			enhanced_img = enhancer.enhance(2)  # Increase contrast

			# Convert to grayscale
			gray_img = ImageOps.grayscale(enhanced_img)
			gray_img_np = np.array(gray_img)
			# Save the processed image to a file for inspection
			processed_image_path = os.path.join(output_folder, f"{keyword.replace(' ', '_')}_enhance.png")
			gray_img = Image.fromarray(gray_img_np)

			gray_img.save(processed_image_path)

			# Run OCR on the cropped image and store the result
			crop_ocr_result = reader.readtext(processed_image_path, detail=0, paragraph=True)
			ocr_results[keyword.title()] = ' '.join(crop_ocr_result)
	return ocr_results

# Example usage:
# Replace 'your_image.jpg' with the path to your image file
image_path = '/media/greydon/lhsc_data/datasets/SEEG/derivatives/frame-based_files/sub-F002/renishaw_screenshots/LAm.png'
additional_keywords = ['OtherKeyword1', 'OtherKeyword2']  # Add any additional keywords you want to search for

# Call the function and save the cropped images
saved_crops = read_text_and_save_crops(image_path, additional_keywords=additional_keywords)
print(saved_crops)



png = Image.open(image_path).convert('RGBA')
alpha = png.convert('RGBA').split()[-1]
background = Image.new('RGBA', png.size, (255,255,255))
background.paste(png, mask=alpha)
alpha_composite = Image.alpha_composite(background, png)
background.save(os.path.splitext(image_path)[0]+'_.png', 'PNG', quality=150)

with open(os.path.splitext(image_path)[0]+'.pdf', "wb") as f:
	f.write(img2pdf.convert(os.path.splitext(image_path)[0]+'_.png'))

import ocrmypdf

ocrmypdf.ocr(os.path.splitext(image_path)[0]+'.pdf', os.path.splitext(image_path)[0]+'.pdf', skip_text=False)
import cv2
import pytesseract


image = cv2.imread('/media/greydon/lhsc_data/datasets/SEEG/derivatives/frame-based_files/sub-F002/renishaw_screenshots/content/target_enhance.png')
thresh = cv2.threshold(image, 0, 255, 100)[1]
cv2.imshow('thresh', thresh)

data = pytesseract.image_to_string(image, lang='eng',config='--psm 3')
print(data)



pdf = pdfplumber.open(os.path.splitext(image_path)[0]+'.pdf')

all_text = ''
with pdfplumber.open(os.path.splitext(image_path)[0]+'.pdf') as pdf:
	for pdf_page in pdf.pages:
		single_page_text = pdf_page.extract_text()
		all_text = all_text + ' ' + single_page_text

all_text=all_text.replace('\n',' ')


for line in text.split('\n'):
    if '/' in line:
        line = line.split('/')[1].split(' ')[0]
        print(line)
		
		