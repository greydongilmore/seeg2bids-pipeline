#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 13:24:58 2023

@author: greydon
"""

from PIL import Image, ImageEnhance, ImageOps,ImageFilter
import easyocr
import ocrmypdf
import cv2
import os
import img2pdf
import numpy as np
import pdfplumber
import numpy as np
import regex as re
import glob
import os
import shutil
import pandas as pd
from tempfile import mkdtemp
import skimage.morphology as morph 


regEnv = os.environ.copy()
regEnv['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

def run_command(cmdLineArguments, regEnv=None):
	if regEnv is not None:
		subprocess.run(cmdLineArguments, stdout=subprocess.PIPE,stderr=subprocess.STDOUT, shell=True, env=(regEnv))
	else:
		subprocess.run(cmdLineArguments, stdout=subprocess.PIPE,stderr=subprocess.STDOUT, shell=True)

def read_coords(image):
	reader = easyocr.Reader(['en'])
	results = reader.readtext(image)
	return results

def read_text_and_save_crops(out_fname, keywords=None):
	
	output_folder=os.path.join(os.path.dirname(out_fname),'content')
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	
	img = Image.open(out_fname)
	keyword_bboxes = {k: None for k in list(keywords)}
	
	reader = easyocr.Reader(['en'],gpu=True)
	results = reader.readtext(out_fname, low_text=0.4)
	
	for keyword in list(keywords):
		texts=[x[1] for x in results]
		idxs=indices(texts, keyword.title())
		weight=[x for x in idxs if x != idxs[np.argmax(np.array([x[2] for x in results])[idxs])]]
		for i in reversed(weight):
			del results[i]
	
	for (bbox, text, prob) in results:
		text = text.lower()
		for keyword in list(keywords):
			if keyword == text:
				expansion_x = (bbox[2][0] - bbox[0][0]) * keywords[keyword]['expansion']
				expansion_y = (bbox[2][1] - bbox[0][1]) * keywords[keyword]['expansion']
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
	
	ocr_results = {}
	saved_crops = {}
	for keyword, bbox in keyword_bboxes.items():
		if bbox:
			file_name = f"{keyword.replace(' ', '_')}.png"
			crop_path = os.path.join(output_folder, file_name)
			save_crop(img, bbox, crop_path)
			saved_crops[keyword.title()] = crop_path
			
			crop_img = Image.open(crop_path)
			crop_img_morph = morph.erosion(crop_img,np.ones((1, 1)))
			cv2.imwrite(crop_path, crop_img_morph)
			
			# Run OCR on the cropped image and store the result
			reader = easyocr.Reader(['en'],gpu=True)
			crop_ocr_result = reader.readtext(crop_path, low_text=0.3,detail=0,width_ths=0.7,text_threshold =0.4,adjust_contrast=.7)
			
			if keywords[keyword]['type']=='num':
				ocr_results[keyword.title()] = np.array(re.findall(r"[-+]?(?:\d*\.*\d+)", ' '.join(crop_ocr_result).replace(".,",'.').replace(",",'.').replace("..",'.').replace("~",'-'))).astype(float).tolist()
				if keywords[keyword]['filter']:
					for idx,ifloat in enumerate(ocr_results[keyword.title()]):
						if abs(ifloat)>100:
							ocr_results[keyword.title()][idx]=(ifloat/10)
			elif keywords[keyword]['type']=='text':
				ocr_results[keyword.title()] = [''.join(filter(lambda x: not x.isdigit(), ' '.join(crop_ocr_result))).replace(",",'').replace("~",'').replace(".",'')]
			
			for iword in keywords[keyword]['ignore']:
				ocr_results[keyword.title()]=[x.replace(iword,'').strip() for x in ocr_results[keyword.title()]][0]
	
	return ocr_results

def save_crop(img, bbox, file_name):
	cropped_img = img.crop(bbox)  # Crop the image to the bounding box
	cropped_img.save(file_name)   # Save the cropped image as a PNG file

def indices(lst, item):
	return [i for i, x in enumerate(lst) if x == item]

def writeFCSV(coords,labels,descriptions,output_fcsv=None,coordsys='0'):
	
	with open(output_fcsv, 'w') as fid:
		fid.write("# Markups fiducial file version = 4.11\n")
		fid.write(f"# CoordinateSystem = {coordsys}\n")
		fid.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
	
	out_df={'node_id':[],'x':[],'y':[],'z':[],'ow':[],'ox':[],'oy':[],'oz':[],
		'vis':[],'sel':[],'lock':[],'label':[],'description':[],'associatedNodeID':[]
	}
	if len(labels)<1:
		labels=np.repeat("",len(coords))
	if len(descriptions)<1:
		descriptions=np.repeat("",len(coords))
	
	for ilabels,idesc,icoords,idx in zip(labels,descriptions,coords,range(len(coords))):
		out_df['node_id'].append(idx+1)
		out_df['x'].append(icoords[0])
		out_df['y'].append(icoords[1])
		out_df['z'].append(icoords[2])
		out_df['ow'].append(0)
		out_df['ox'].append(0)
		out_df['oy'].append(0)
		out_df['oz'].append(0)
		out_df['vis'].append(1)
		out_df['sel'].append(1)
		out_df['lock'].append(1)
		out_df['label'].append(ilabels)
		out_df['description'].append(idesc)
		out_df['associatedNodeID'].append('')
	
	out_df=pd.DataFrame(out_df)
	out_df.round(3).to_csv(output_fcsv, sep=',', index=False, lineterminator="", mode='a', header=False)

def sorted_nicely(lst):
	convert = lambda text: int(text) if text.isdigit() else text
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	sorted_lst = sorted(lst, key = alphanum_key)
	
	return sorted_lst

keywords = {
	'trajectory':{
		'expansion':6,
		'type':'text',
		'ignore':['Trajectory'],
		'filter':False
	},
	'target':{
		'expansion':5,
		'type':'num',
		'ignore':[],
		'filter':True
	},
	'trajectory length':{
		'expansion':1,
		'type':'num',
		'ignore':[],
		'filter':False
	}
}
def get_ROI(image, horizontal, vertical, left_line_index, right_line_index, top_line_index, bottom_line_index, offset=4):
	x1 = vertical[left_line_index][2] + offset
	y1 = horizontal[top_line_index][3] + offset
	x2 = vertical[right_line_index][2] - offset
	y2 = horizontal[bottom_line_index][3] - offset
	
	w = x2 - x1
	h = y2 - y1
	
	cropped_image = get_cropped_image(image, x1, y1, w, h)
	
	return cropped_image, (x1, y1, w, h)
def get_cropped_image(image, x, y, w, h):
	cropped_image = image[ y:y+h , x:x+w ]
	return cropped_image
def is_vertical(line):
	return line[0]==line[2]

def is_horizontal(line):
	return line[1]==line[3]
def overlapping_filter(lines, sorting_index):
	filtered_lines = []
	
	lines = sorted(lines, key=lambda lines: lines[sorting_index])
	
	for i in range(len(lines)):
			l_curr = lines[i]
			if(i>0):
				l_prev = lines[i-1]
				if ( (l_curr[sorting_index] - l_prev[sorting_index]) > 5):
					filtered_lines.append(l_curr)
			else:
				filtered_lines.append(l_curr)
				
	return filtered_lines

def angle_cos(p0, p1, p2):
	d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
	return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(image, threshold):
	image = cv2.GaussianBlur(image, (5, 5), 0)
	squares = []
	for gray in cv2.split(image):
		for thrs in range(0, 255, 26):
			if thrs == 0:
				bin = cv2.Canny(gray, 0, 50, apertureSize=5)
				bin = cv2.dilate(bin, None)
			else:
				_retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
			contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			for cnt in contours:
				cnt_len = cv2.arcLength(cnt, True)
				cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
				if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
					cnt = cnt.reshape(-1, 2)
					max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
					if max_cos < threshold:
						squares.append(cnt)
	return squares
def crop_image(im,squares, padding=[0,0,0,0]):
	# top, right, bottom, left
	
	# Setting the points for cropped image
	left, top = squares[0][0]-padding[3], squares[0][1]-padding[0]
	right, bottom = squares[1][0]+padding[1], squares[1][1]+padding[2]
	
	# Cropped image of above dimension
	# (It will not change orginal image)
	im1 = im.crop((left, top, right, bottom))

	filename = './temp/' + str(i) + '.png'
	im1.save(filename)
	
	return filename
	
	# Shows the image in image viewer
	# im1.show()

def get_cords(squares,pos='Top',threshold=50):
	if pos=='Top':
		idx=0
	elif pos=='Bot':
		idx=2
	
	top_x = []
	for i in squares:
		top_x.append(i[idx][0])
	top_x = np.unique(top_x)
	
	x_bins = [top_x[0]]
	for i in top_x:
		if i - x_bins[-1] > threshold:
			x_bins.append(i)
	
	top_y = []
	for i in squares:
		top_y.append(i[idx][1])
	top_y = np.unique(top_y)
	
	y_bins = [top_y[0]]
	for i in top_y:
		if i - y_bins[-1] > threshold:
			y_bins.append(i)
	
	print(x_bins)
	print(y_bins)
	
	if pos=='Top':
		top_cords = []
		for i in x_bins[1:-1]:
			for j in y_bins[1:]:
				top_cords.append((i,j))
		return top_cords
	elif pos=='Bot':
		bottom_cords = []
		for i in x_bins[1:]:
			if i != 0:
				for j in y_bins[:-1]:
					if j != 0:
						bottom_cords.append((i,j))
		return bottom_cords

#%%

import imutils,pytesseract,string,subprocess,tabula



image_path = r"E:\datasets\SEEG\derivatives\frame-based_files"

for participant in sorted_nicely([x for x in os.listdir(image_path) if x.startswith('sub') and '_ses-' not in x]):
	
	tmpdir = mkdtemp(prefix='renishaw')
	final_coords=[]
	for iimg in glob.glob(os.path.join(image_path,participant,'**',"*.png")):
		out_fname= os.path.join(tmpdir, os.path.splitext(os.path.basename(iimg))[0]+'_morph.png')
		
		#if not os.path.exists(os.path.join(os.path.dirname(iimg), 'extract', os.path.basename(out_fname))):
		image = cv2.imread(iimg)
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		adaptive = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
		
		cnts = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		squares = find_squares(gray_image, 0.1)
		cv2.drawContours(image, squares, -1, (0,255,0),3)
		top_cords=get_cords(squares,'Top',50)
		bottom_cords=get_cords(squares,'Bot',1)
		
		fig, ax = plt.subplots(figsize=(16,12))
		ax.imshow(image[::-1],origin='lower')
		ax.invert_yaxis()
		im = Image.open(iimg)

		all_cords = list(zip(top_cords, bottom_cords))
		for i in range(len(all_cords)):
			crop_image(im,all_cords[i], padding=[2,2,2,2])
	
		(thresh, blackAndWhiteImage) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
		
		opening=cv2.morphologyEx(blackAndWhiteImage, cv2.MORPH_OPEN,  np.ones((2,2),np.uint8))
		opening=cv2.morphologyEx(opening, cv2.MORPH_ERODE,  np.ones((1,1),np.uint8))
		
		cv2.imwrite(out_fname, gray_image)
		
		layout= img2pdf.get_fixed_dpi_layout_fun((350, 350))
		with open(os.path.splitext(out_fname)[0]+".pdf", "wb") as f:
			f.write(img2pdf.convert(out_fname, layout_fun=layout))
		
		ocrmypdf.ocr(os.path.splitext(out_fname)[0]+".pdf",os.path.splitext(out_fname)[0]+"_.pdf",language=['eng'],output_type='pdf',clean=True,deskew=True)
		
		commands = ['gswin64.exe', '-dPDFA', '-dBATCH', '-dNOPAUSE', '-sColorConversionStrategy=UseDeviceIndependentColor', '-sProcessColorModel=DeviceCMYK',\
			  '-sDEVICE=pdfwrite', '-sPDFACompatibilityPolicy=1', '-sOutputFile=' + os.path.splitext(out_fname)[0]+"_ocr.pdf", os.path.splitext(out_fname)[0]+"_.pdf"]
		
		ocr_cmd=' '.join(commands)
		run_command(ocr_cmd,regEnv)
		
		frame_info = tabula.read_pdf(os.path.splitext(out_fname)[0]+".pdf", encoding='latin1', pages='1', area=[270,65,390,240])[0]
		frame_idx=[i for i,x in enumerate(frame_info.iloc[:,0].fillna('')) if 'Scanned' in x]
		
		
		img = Image.open(out_fname)
		
		fig, ax = plt.subplots(figsize=(16,12))
		ax.imshow(blackAndWhiteImage)
		
		fig, ax = plt.subplots(figsize=(16,12))
		ax.imshow(opening)
		
		for i in top_cords:
			ax.scatter([i[0]], [i[1]], marker='x', c='r')
		for i in bottom_cords:
			ax.scatter([i[0]], [i[1]], marker='o', c='g')
		
		
		contours = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = contours[0] if len(contours) == 2 else contours[1]
		cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		tableCnt = max(cnts, key=cv2.contourArea)
		(x, y, w, h) = cv2.boundingRect(tableCnt)
		table = image[y:y + h, x:x + w]
		options = "--psm 6"
		results = pytesseract.image_to_data(table,config=options,output_type=pytesseract.Output.DICT)
		
		plt.imshow(opening)
		
		custom_config = r"--psm 6 -c tessedit_char_whitelist= '0123456789-'"
		text = pytesseract.image_to_string(table, lang='eng', config=custom_config)
		print(text)

		fig, ax = plt.subplots(figsize=(16,12))
		ax.imshow(table)
		ax[1].imshow(img_erosion)
		plt.tight_layout()


		#else:
		#	source_name=os.path.join(os.path.dirname(iimg), 'extract', os.path.basename(out_fname))
		#	shutil.copy2(source_name, out_fname)
		
		ocr_results=read_text_and_save_crops(out_fname,keywords)
		ocr_results['Label']=os.path.splitext(os.path.basename(iimg))[0]
		if len(ocr_results['Target'])==6:
			ocr_results['Entry']=(np.array(ocr_results['Target'][1::2])*np.array([-1,-1,1])).tolist()
			ocr_results['Target']=(np.array(ocr_results['Target'][0::2])*np.array([-1,-1,1])).tolist()
		
		if all(x in list(ocr_results) for x in ('Entry','Target')):
			for ipoint in ('Entry','Target'):
				for idx,ifloat in enumerate(ocr_results[ipoint]):
					if abs(ifloat)>100:
						ocr_results[ipoint][idx]=(ifloat/10)
		final_coords.append(ocr_results)
	
	out_fcsv=os.path.join(os.path.dirname(image_path),'planned_coords',participant,f'{participant}_planned.fcsv')
	if not os.path.exists(os.path.dirname(out_fcsv)):
		os.makedirs(os.path.dirname(out_fcsv))
	
	if final_coords:
		final_coords=pd.DataFrame(final_coords)
		coords=np.vstack(([np.stack((x['Target'],x['Entry'])) for i,x in final_coords.iterrows()]))
		descriptions=np.vstack(([np.vstack(np.repeat(x['Trajectory'],2)) for i,x in final_coords.iterrows()])).flatten()
		labels=np.vstack(([np.vstack(np.repeat(x['Label'],2)) for i,x in final_coords.iterrows()])).flatten()
		writeFCSV(coords,labels,descriptions,output_fcsv=out_fcsv,coordsys='0')
	
	out_pngs=os.path.join(os.path.dirname(iimg),'extract')
	if not os.path.exists(os.path.dirname(out_pngs)):
		os.makedirs(os.path.dirname(out_pngs))
		
	for file_name in glob.glob(os.path.join(tmpdir,"*")):
		if os.path.isfile(file_name):
			shutil.move(file_name, out_pngs)


all_text = ''
with pdfplumber.open(os.path.splitext(out_fname)[0]+"_.pdf") as pdf:
	for pdf_page in pdf.pages:
		single_page_text = pdf_page.extract_text()
		all_text = all_text + ' ' + single_page_text

all_text=all_text.replace('\n',' ')


for line in text.split('\n'):
    if '/' in line:
        line = line.split('/')[1].split(' ')[0]
        print(line)
		
		