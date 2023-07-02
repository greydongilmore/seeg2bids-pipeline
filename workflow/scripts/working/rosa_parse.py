#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 22:13:00 2023

@author: greydon
"""

import numpy as np
import regex as re
from functools import reduce
import nibabel as nb
import glob
import os
import pandas as pd
from collections import ChainMap

def extractTokens(textfile):
	# Token starts with [%%tokenname%%]
	# Tokens are in a line
	# Ends before the next []
	tokens = []
	starti = [m.start() for m in re.finditer(r'\[(.*?)\]', textfile)]
	endi = [m.end() for m in re.finditer(r'\[(.*?)\]', textfile)]
	
	for i in range(len(starti) - 1):  # Last token should be [END]
		token={}
		token[textfile[starti[i]+1:endi[i]-1]] = textfile[endi[i]+1:starti[i+1]-1]
		tokens.append(token)
	return tokens


def parseROSAfile(ros_fname):
	rosadata = {
		'ATFormRAS': np.diag([-1,-1,1,1]),
	}
	
	with open(ros_fname, 'r') as f:
		textfile = f.read()
	
	rosadata['ac']=getCoordinates(textfile, 'AC').tolist()
	rosadata['pc']=getCoordinates(textfile, 'PC').tolist()
	rosadata['ih']=getCoordinates(textfile, 'IH').tolist()
	rosadata['trajectories']=getTrajectoriesList(textfile)
	rosadata['volumes']=getVolumes(textfile)
	
	return rosadata

def getVolumes(textfile):
	displays=[]
	result = re.findall(r'\[IMAGERY_TYPE\]\n(.+?)\n\[IMAGERY_3DREF\]', textfile, re.DOTALL)
	for iresult in result:
		data = dict(ChainMap(*extractTokens(iresult)))
		for key,value in data.items():
			temp=value.split('\n')[-1].strip()
			if key == 'TRdicomRdisplay':
				temp=np.array([float(x) for x in temp.split(' ')]).reshape(4, 4)
			data[key]=temp
		displays.append(data)
	
	return displays

def getCoordinates(textfile, queryPoint):
	pattern = r"(?<=\[ACPC\]).*" + queryPoint + r" \d -?\d+\.\d+ -?\d+\.\d+ -?\d+\.\d+"
	m = re.search(pattern, textfile, re.DOTALL)
	coords_str = m.group().split(' ')[-3:]
	coords_lps = np.array(list(map(float, coords_str)))
	coords_ras = coords_lps * np.array([-1, -1, 1])
	return coords_ras

def getTrajectoriesList(textfile):
	pattern = r"(?P<name>[\^-\w]+) (?P<type>\d) (?P<color>\d+) (?P<entry_point_defined>\d) (?P<entry>-?\d+\.\d+ -?\d+\.\d+ -?\d+\.\d+) (?P<target_point_defined>\d) (?P<target>-?\d+\.\d+ -?\d+\.\d+ -?\d+\.\d+) (?P<instrument_length>\d+\.\d+) (?P<instrument_diameter>\d+\.\d+)\n"
	trajectories = [m.groupdict() for m in re.finditer(pattern, textfile)]
	for trajectory in trajectories:
		trajectory['name']=trajectory['name'].replace('^',' ')
		for pos in ['entry', 'target']:
			trajectory[pos] = np.array(list(map(float, trajectory[pos].split(' ')))) # str to array
			trajectory[pos] = np.round(trajectory[pos] * np.array([-1, -1, 1]),3) # LPS to RAS
	return trajectories

def writeFCSV(coords,labels,output_fcsv):
	
	with open(output_fcsv, 'w') as fid:
		fid.write("# Markups fiducial file version = 4.11\n")
		fid.write("# CoordinateSystem = 0\n")
		fid.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
	
	out_df={'node_id':[],'x':[],'y':[],'z':[],'ow':[],'ox':[],'oy':[],'oz':[],
		'vis':[],'sel':[],'lock':[],'label':[],'description':[],'associatedNodeID':[]
	}
	
	for ilabels,icoords,idx in zip(labels,coords,range(len(coords))):
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
		out_df['description'].append('')
		out_df['associatedNodeID'].append('')
	
	out_df=pd.DataFrame(out_df)
	out_df.round(3).to_csv(output_fcsv, sep=',', index=False, lineterminator="", mode='a', header=False)


#%%


ros_file_path=r'/home/greydon/Documents/data/SEEG_peds/derivatives/seeg_scenes'


for isub in sorted([x for x in os.listdir(ros_file_path) if x.startswith('sub-')])[:1]:
	nii_fname=glob.glob(f"{ros_file_path}/{isub}/*-contrast*_T1w.nii.gz")
	ros_fname=glob.glob(f"{ros_file_path}/{isub}/**/*.ros")
	out_fcsv=os.path.join(ros_file_path,isub,f'{isub}_planned.fcsv')
	out_tfm=os.path.join(ros_file_path,isub,f'{isub}_planned.tfm')
	
	if nii_fname and ros_fname and not os.path.exists(out_fcsv):
		#centering transform
		orig_nifti=nb.load(nii_fname[0])
		center_coordinates=np.array([x/ 2 for x in orig_nifti.header["dim"][1:4]])-1.0
		homogeneous_coord = np.concatenate((center_coordinates, np.array([1])), axis=0)
		centering_transform=np.c_[np.r_[np.eye(3),np.zeros((1,3))], np.dot(orig_nifti.affine,homogeneous_coord)]
		
		rosa_parsed=parseROSAfile(ros_fname[0])
		
		Parameters = " ".join([str(x) for x in np.concatenate((centering_transform[0:3,0:3].reshape(9), centering_transform[0:3,3]))])
		#output_matrix_txt = filen.split('.txt')[0] + '.tfm'
		
		with open(out_tfm, 'w') as fid:
			fid.write("#Insight Transform File V1.0\n")
			fid.write("#Transform 0\n")
			fid.write("Transform: AffineTransform_double_3_3\n")
			fid.write("Parameters: " + Parameters + "\n")
			fid.write("FixedParameters: 0 0 0\n")
		
		coords=[]
		labels=[]
		for idx,traj in enumerate(rosa_parsed['trajectories']):
			vec = np.hstack([traj['target'],1])
			tvec = centering_transform @ vec.T
			traj['target_t']=np.round(tvec,3).tolist()[:3]
			coords.append(traj['target_t'])
			labels.append(traj['name'])
			
			vec = np.hstack([traj['entry'],1])
			tvec = centering_transform @ vec.T
			traj['entry_t']=np.round(tvec,3).tolist()[:3]
			coords.append(traj['entry_t'])
			labels.append(traj['name'])
			
			rosa_parsed['trajectories'][idx]=traj
			
			
		writeFCSV(coords,labels,out_fcsv)
		
		print(f"Done {isub}")
	