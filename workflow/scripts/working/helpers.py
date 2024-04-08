#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import csv
from collections import ChainMap
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def determineFCSVCoordSystem(input_fcsv,overwrite_fcsv=False):
	# need to determine if file is in RAS or LPS
	# loop through header to find coordinate system
	coordFlag = re.compile('# CoordinateSystem')
	verFlag = re.compile('# Markups fiducial file version')
	headFlag = re.compile('# columns')
	coord_sys=None
	headFin=None
	ver_fin=None
	
	with open(input_fcsv, 'r+') as fid:
		rdr = csv.DictReader(filter(lambda row: row[0]=='#', fid))
		row_cnt=0
		for row in rdr:
			cleaned_dict={k:v for k,v in row.items()}
			if None in list(cleaned_dict):
				cleaned_dict['# columns'] = cleaned_dict.pop(None)
			
			if any(coordFlag.match(x) for x in list(cleaned_dict.values()) if not isinstance(x,list)):
				coordString = list(filter(coordFlag.match,  list(cleaned_dict.values())))
				assert len(coordString)==1
				coord_sys = coordString[0].split('=')[-1].strip()
			if any(verFlag.match(x) for x in list(cleaned_dict)):
				verString = list(filter(verFlag.match,  list(cleaned_dict)))
				assert len(verString)==1
				ver_fin = verString[0].split('=')[-1].strip()
			if any(headFlag.match(x) for x in list(cleaned_dict)):
				headString = list(filter(headFlag.match, list(cleaned_dict)))
				headFin=['id']+cleaned_dict[headString[0]]
			row_cnt +=1
	
	if any(x in coord_sys for x in {'LPS','1'}):
		df = pd.read_csv(input_fcsv, skiprows=3, header=None)
		
		if df.shape[1] != 13:
			df=df.iloc[:,:14]
		
		df[1] = -1 * df[1] # flip orientation in x
		df[2] = -1 * df[2] # flip orientation in y
		
		if overwrite_fcsv:
			with open(input_fcsv, 'w') as fid:
				fid.write("# Markups fiducial file version = 4.11\n")
				fid.write("# CoordinateSystem = 0\n")
				fid.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
			
			df.rename(columns={0:'node_id', 1:'x', 2:'y', 3:'z', 4:'ow', 5:'ox',
								6:'oy', 7:'oz', 8:'vis', 9:'sel', 10:'lock',
								11:'label', 12:'description', 13:'associatedNodeID'}, inplace=True)
			
			df['associatedNodeID']= pd.Series(np.repeat('',df.shape[0]))
			df['label']=[x.strip() for x in df['label']]
			df.round(6).to_csv(input_fcsv, sep=',', index=False, lineterminator="", mode='a', header=False, float_format='%.6f')
			
			print(f"Converted LPS to RAS: {os.path.dirname(input_fcsv)}/{os.path.basename(input_fcsv)}")
	return coord_sys,headFin

def determine_groups(iterable, numbered_labels=False):
	values = []
	for item in iterable:
		item=item.strip()
		temp=None
		if re.findall(r"([a-zA-Z]+)([0-9]+)([a-zA-Z]+)", item):
			temp = "".join(list(re.findall(r"([a-zA-Z]+)([0-9]+)([a-zA-Z]+)", item)[0]))
		elif '-' in item:
			temp=item.split('-')[0]
		else:
			if numbered_labels:
				temp=''.join([x for x in item if not x.isdigit()])
				for sub in ("T1","T2"):
					if sub in item:
						temp=item.split(sub)[0] + sub
			else:
				temp=item
		if temp is None:
			temp=item
		
		values.append(temp)
	
	vals,indexes,count = np.unique(values, return_index=True, return_counts=True)
	values_unique = [values[index] for index in sorted(indexes)]
	
	return values_unique,vals

def sorted_nicely(data, reverse = False):
	convert = lambda text: int(text) if text.isdigit() else text
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	
	return sorted(data, key = alphanum_key, reverse=reverse)

def normDir(direction):
	return np.array(direction) / np.linalg.norm(np.array(direction))

def mag_vec(P1, P2):
	if isinstance(P1, list):
		P1 = np.array(P1)
	if isinstance(P1, list):
		P2 = np.array(P2)
	DirVec = P2-P1
	MagVec = np.sqrt([np.square(DirVec[0]) + np.square(DirVec[1]) + np.square(DirVec[2])])
	return MagVec

def norm_vec(P1, P2):
	if isinstance(P1, list):
		P1 = np.array(P1)
	if isinstance(P2, list):
		P2 = np.array(P2)
	DirVec = P2-P1
	MagVec = np.sqrt([np.square(DirVec[0]) + np.square(DirVec[1]) + np.square(DirVec[2])])
	NormVec = np.array([float(DirVec[0] / MagVec), float(DirVec[1] / MagVec), float(DirVec[2] / MagVec)])
	return NormVec

def writeFCSV(data_fcsv, fcsv_fname, coord_sys='RAS'):
	with open(fcsv_fname, 'w') as fid:
		fid.write("# Markups fiducial file version = 4.11\n")
		fid.write(f"# CoordinateSystem = {coord_sys}\n")
		fid.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
	
	data_fcsv.to_csv(fcsv_fname, sep=',', encoding='utf-8', header= False, index = False, na_rep='', mode='a', lineterminator="", float_format='%.6f')

class IndexTracker:
	def __init__(self, ax, img_data,points=None, rotate_img=False, rotate_points=False, title=None):
		self.ax = ax
		self.scatter=None
		self.points=None
		self.title=title
		no_index= True
		if rotate_img:
			self.img_data = np.fliplr(np.rot90(img_data.copy(),3))
			#if self.points is not None:
			#	self.points = np.rot90(points, k=1)
		else:
			self.img_data = img_data.copy()
		
		rows, cols, self.slices = self.img_data.shape
		self.ind = self.slices//2
		if points is not None:
			self.points=points.copy()
			if rotate_points:
				self.points[:,[0,1]]=self.points[:,[1,0]]
			while no_index==True:
				if any(self.points[:,2]==self.ind):
					no_index=False
					point_plot=np.vstack([np.mean(self.points[(self.points[:,2]==self.ind)*(self.points[:,3]==x),:2],0) for x in np.unique(self.points[self.points[:,2]==self.ind,3])])
					self.scatter,=ax.plot(point_plot[:,1],point_plot[:,0], marker="o", markersize=12, c = 'yellow', fillstyle='none', markeredgewidth=1, linestyle = 'None')
				else:
					self.ind+=1
				
		self.im = ax.imshow(self.img_data[:, :, self.ind], origin='lower')
		self.update()
	
	def on_scroll(self, event):
		print("%s %s" % (event.button, event.step))
		if event.button == 'up':
			self.ind = (self.ind + 1) % self.slices
		else:
			self.ind = (self.ind - 1) % self.slices
		self.update()
	
	def update(self):
		
		if self.points is not None:
			if any(self.points[:,2]==self.ind):
				point_plot=np.vstack([np.mean(self.points[(self.points[:,2]==self.ind)*(self.points[:,3]==x),:2],0) for x in np.unique(self.points[self.points[:,2]==self.ind,3])])
				self.scatter.set_xdata(point_plot[:,1])
				self.scatter.set_ydata(point_plot[:,0])
		self.im.set_data(self.img_data[:, :, self.ind])
		
		if self.title is not None:
			plot_title=self.title
		else:
			plot_title='slice %s' % self.ind
		
		self.ax.set_title(plot_title, fontdict={'fontsize': 18, 'fontweight': 'bold'})
		self.ax.tick_params(axis='x', labelsize=14)
		self.ax.tick_params(axis='y', labelsize=14)
		self.im.axes.figure.canvas.draw()

#%%

electrodeModels = {}

RD10RSP03 = {
		'num_groups': 10,
		'num_contacts': 10,
		'encapsultation': 1.0,
		'lead_shift':1.5,
		'lead_tail': 15,
		'contact_size': 2.29,
		'contact_spacing': [0.71,0.71,0.71,0.71,0.71,0.71,0.71,0.71,0.71],
		'diameter': 0.86,
		'electrode_1': [1,2,3,4,5,6,7,8,9,10],
		'electrode_2': [1,2,3,4,5,6,7,8,9,10],
		'contact_label':['','','','','','','', ''],
		'lead_type': 'linear',
		'filename': 'rd10rsp03',
		'catalog_num': 'RD10R-SP03X'
	 }
	 
electrodeModels['RD10RSP03'] = RD10RSP03

RD10RSP04 = {
		'num_groups': 10,
		'num_contacts': 10,
		'encapsultation': 1.0,
		'lead_shift':1.5,
		'lead_tail': 15,
		'contact_size': 2.29,
		'contact_spacing': [1.71,1.71,1.71,1.71,1.71,1.71,1.71,1.71,1.71],
		'diameter': 0.86,
		'electrode_1': [1,2,3,4,5,6,7,8,9,10],
		'electrode_2': [1,2,3,4,5,6,7,8,9,10],
		'contact_label':['','','','','','','', ''],
		'lead_type': 'linear',
		'filename': 'rd10rsp04',
		'catalog_num': 'RD10R-SP04X'
	 }
	 
electrodeModels['RD10RSP04'] = RD10RSP04

RD10RSP05 = {
		'num_groups': 10,
		'num_contacts': 10,
		'encapsultation': 1.0,
		'lead_shift':1.5,
		'lead_tail': 15,
		'contact_size': 2.29,
		'contact_spacing': [2.71,2.71,2.71,2.71,2.71,2.71,2.71,2.71,2.71],
		'diameter': 0.86,
		'electrode_1': [1,2,3,4,5,6,7,8,9,10],
		'electrode_2': [1,2,3,4,5,6,7,8,9,10],
		'contact_label':['','','','','','','', ''],
		'lead_type': 'linear',
		'filename': 'rd10rsp05',
		'catalog_num': 'RD10R-SP05X'
	 }
	 
electrodeModels['RD10RSP05'] = RD10RSP05

RD10RSP06 = {
		'num_groups': 10,
		'num_contacts': 10,
		'encapsultation': 1.0,
		'lead_shift':1.5,
		'lead_tail': 15,
		'contact_size': 2.29,
		'contact_spacing':[3.71,3.71,3.71,3.71,3.71,3.71,3.71,3.71,3.71],
		'diameter': 0.86,
		'electrode_1': [1,2,3,4,5,6,7,8,9,10],
		'electrode_2': [1,2,3,4,5,6,7,8,9,10],
		'contact_label':['','','','','','','', ''],
		'lead_type': 'linear',
		'filename': 'rd10rsp06',
		'catalog_num': 'RD10R-SP06X'
	 }
	 
electrodeModels['RD10RSP06'] = RD10RSP06

RD10RSP07 = {
		'num_groups': 10,
		'num_contacts': 10,
		'encapsultation': 1.0,
		'lead_shift':1.5,
		'lead_tail': 15,
		'contact_size': 2.29,
		'contact_spacing': [4.71,4.71,4.71,4.71,4.71,4.71,4.71,4.71,4.71],
		'diameter': 0.86,
		'electrode_1': [1,2,3,4,5,6,7,8,9,10],
		'electrode_2': [1,2,3,4,5,6,7,8,9,10],
		'contact_label':['','','','','','','', ''],
		'lead_type': 'linear',
		'filename': 'rd10rsp07',
		'catalog_num': 'RD10R-SP07X'
	 }
	 
electrodeModels['RD10RSP07'] = RD10RSP07

MM16DSP05 = {
		'num_groups': 8,
		'num_contacts': 8,
		'encapsultation': 1.5,
		'lead_shift':1.5,
		'lead_tail': 15,
		'contact_size': 2.29,
		'contact_spacing': [2.71,2.71,2.71,2.71,2.71,2.71,2.71,2.71,2.71],
		'diameter': 1.3,
		'electrode_1': [1,2,3,4,5,6,7,8],
		'electrode_2': [1,2,3,4,5,6,7,8],
		'contact_label':['','','','','','','', ''],
		'lead_type': 'linear',
		'filename': 'mm16dsp05'
	 }
	 
electrodeModels['MM16D-SP05X'] = MM16DSP05

BF09RSP51X = {
		'num_groups': 9,
		'num_contacts': 9,
		'encapsultation': 1.5,
		'lead_shift':1.5,
		'lead_tail': 15,
		'contact_size': 1.6,
		'contact_spacing': [1.4, 3.4, 3.4, 3.4, 3.4, 3.4, 3.4, 3.4],
		'diameter': 1.3,
		'electrode_1': [1,2,3,4,5,6,7,8],
		'electrode_2': [1,2,3,4,5,6,7,8],
		'contact_label':['','','','','','','', ''],
		'lead_type': 'linear',
		'filename': 'bf09rsp51x'
	 }
	 
electrodeModels['BF09R-SP51X'] = BF09RSP51X

D0805AM = {
		'num_groups': 5,
		'num_contacts': 5,
		'encapsultation': 0,
		'lead_shift':0,
		'lead_tail': 15,
		'contact_size': 2.0,
		'contact_spacing': [1.5, 1.5, 1.5, 1.5],
		'diameter': 0.8,
		'electrode_1': [1,2,3,4,5],
		'electrode_2': [1,2,3,4,5],
		'contact_label':['','','','',''],
		'lead_type': 'linear',
		'filename': 'd0805am'
	 }

electrodeModels['5 contact'] = D0805AM

D0808AM = {
		'num_groups': 8,
		'num_contacts': 8,
		'encapsultation': 0,
		'lead_shift':0,
		'lead_tail': 15,
		'contact_size': 2.0,
		'contact_spacing': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
		'diameter': 0.8,
		'electrode_1': [1,2,3,4,5,6,7,8],
		'electrode_2': [1,2,3,4,5,6,7,8],
		'contact_label':['','','','','','','',''],
		'lead_type': 'linear',
		'filename': 'd0808am'
	 }

electrodeModels['8 contact'] = D0808AM

D0810AM = {
		'num_groups': 10,
		'num_contacts': 10,
		'encapsultation': 0,
		'lead_shift':0,
		'lead_tail': 15,
		'contact_size': 2.0,
		'contact_spacing': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
		'diameter': 0.8,
		'electrode_1': [1,2,3,4,5,6,7,8,9,10],
		'electrode_2': [1,2,3,4,5,6,7,8,9,10],
		'contact_label':['','','','','','','','','',''],
		'lead_type': 'linear',
		'filename': 'd0810am'
	 }

electrodeModels['10 contact'] = D0810AM

D0812AM = {
		'num_groups': 12,
		'num_contacts': 12,
		'encapsultation': 0,
		'lead_shift':0,
		'lead_tail': 15,
		'contact_size': 2.0,
		'contact_spacing': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
		'diameter': 0.8,
		'electrode_1': [1,2,3,4,5,6,7,8,9,10,11,12],
		'electrode_2': [1,2,3,4,5,6,7,8,9,10,11,12],
		'contact_label':['','','','','','','','','','','',''],
		'lead_type': 'linear',
		'filename': 'd0812am'
	 }

electrodeModels['12 contact'] = D0812AM

D0815AM = {
		'num_groups': 15,
		'num_contacts': 15,
		'encapsultation': 0,
		'lead_shift':0,
		'lead_tail': 15,
		'contact_size': 2.0,
		'contact_spacing': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
		'diameter': 0.8,
		'electrode_1': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
		'electrode_2': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
		'contact_label':['','','','','','','','','','','','','','',''],
		'lead_type': 'linear',
		'filename': 'd0815am'
	 }

electrodeModels['15 contact'] = D0815AM

D0818AM = {
		'num_groups': 18,
		'num_contacts': 18,
		'encapsultation': 0,
		'lead_shift':0,
		'lead_tail': 18,
		'contact_size': 2.0,
		'contact_spacing': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
		'diameter': 0.8,
		'electrode_1': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
		'electrode_2': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
		'contact_label':['','','','','','','','','','','','','','','','','',''],
		'lead_type': 'linear',
		'filename': 'd0818am'
	 }

electrodeModels['18 contact'] = D0818AM
