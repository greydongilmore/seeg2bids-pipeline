#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 23:22:42 2024

@author: greydon
"""
import os,vtk,math,glob
import pandas as pd
import numpy as np
import re
import csv
from bids.layout import BIDSLayout
import shutil
from collections import OrderedDict
from collections import ChainMap

def make_bids_filename(subject_id, space_id, desc_id, suffix, prefix):
	order = OrderedDict([('space', space_id if space_id is not None else None),
						 ('desc', desc_id if desc_id is not None else None)])
	filename = []
	if subject_id is not None:
		filename.append(subject_id)
	for key, val in order.items():
		if val is not None:
			filename.append('%s-%s' % (key, val))
	if isinstance(suffix, str):
		filename.append(suffix)
	filename = '_'.join(filename)
	if isinstance(prefix, str):
		filename = os.path.join(prefix, filename)
		
	return filename

def determineFCSVCoordSystem(input_fcsv):
	# need to determine if file is in RAS or LPS
	# loop through header to find coordinate system
	coordFlag = re.compile('# CoordinateSystem')
	verFlag = re.compile('# Markups fiducial file version')
	headFlag = re.compile('# columns')
	coord_sys=None
	headFin=None
	ver_fin=None
	
	with open(input_fcsv, 'r') as myfile:
		firstNlines=myfile.readlines()[0:3]
	
	for row in firstNlines:
		row=re.sub("[\s\,]+[\,]","",row).replace("\n","")
		cleaned_dict={row.split('=')[0].strip():row.split('=')[1].strip()}
		if None in list(cleaned_dict):
			cleaned_dict['# columns'] = cleaned_dict.pop(None)
		if any(coordFlag.match(x) for x in list(cleaned_dict)):
			coord_sys = list(cleaned_dict.values())[0]
		if any(verFlag.match(x) for x in list(cleaned_dict)):
			verString = list(filter(verFlag.match,  list(cleaned_dict)))
			assert len(verString)==1
			ver_fin = verString[0].split('=')[-1].strip()
		if any(headFlag.match(x) for x in list(cleaned_dict)):
			headFin=list(cleaned_dict.values())[0].split(',')
	
	
	return coord_sys,headFin

def acpc_transform(ac_point,pc_point,mid_point):
	pmprime = (ac_point + pc_point) / 2
	vec1 = ac_point - pmprime
	vec2 = mid_point - pmprime
	vec1Mag = np.sqrt(vec1[0] ** 2 + vec1[1] ** 2 + vec1[2] ** 2)
	vec2Mag = np.sqrt(vec2[0] ** 2 + vec2[1] ** 2 + vec2[2] ** 2)
	vec1Unit = vec1 / vec1Mag
	vec2Unit = vec2 / vec2Mag
	yihatprime = vec1Unit
	if pmprime[2] > mid_point[2]:
		xihatprime = np.cross(vec2Unit, vec1Unit)
	else:
		xihatprime = np.cross(vec1Unit, vec2Unit)
	xAxisMag = (xihatprime[0] ** 2 + xihatprime[1] ** 2 + xihatprime[2] ** 2) ** 0.5
	xihatprime = xihatprime / xAxisMag
	zAxis = np.cross(xihatprime, yihatprime)
	zAxisMag = (zAxis[0] ** 2 + zAxis[1] ** 2 + zAxis[2] ** 2) ** 0.5
	zihatprime = zAxis / zAxisMag
	xihat = np.array([1, 0, 0])
	yihat = np.array([0, 1, 0])
	zihat = np.array([0, 0, 1])
	riiprime = np.vstack([np.array([xihatprime.dot(xihat), xihatprime.dot(yihat), xihatprime.dot(zihat),0]),
						 np.array([yihatprime.dot(xihat), yihatprime.dot(yihat), yihatprime.dot(zihat),0]),
						 np.array([zihatprime.dot(xihat), zihatprime.dot(yihat), zihatprime.dot(zihat),0]),
						 np.array([0,0,0,1])])
	return riiprime

isub='sub-027'

patient_output = r"/home/greydon/Documents/data/SEEG_peds/derivatives/seeg_scenes/sub-P027"
#if not os.path.exists(patient_output):
#	os.makedirs(patient_output)

patient_files = []
for dirpath, subdirs, subfiles in os.walk(patient_output):
	for x in subfiles:
		if x.endswith(".fcsv") and not x.startswith('coords'):
			patient_files.append(os.path.join(dirpath, x))
			
nii_fname=glob.glob(os.path.join(patient_output,'*-contrast*_T1w.nii.gz'))
acpc_file = glob.glob(os.path.join(patient_output,'*acpc.fcsv'))

if acpc_file:
orig_nifti=nb.load(nii_fname[0])
orig_affine=orig_nifti.affine.copy()
center_coordinates=np.array([x/ 2 for x in orig_nifti.header["dim"][1:4]])
homogeneous_coord = np.concatenate((center_coordinates, np.array([1])), axis=0)
centering_transform_raw=np.c_[np.vstack([np.eye(3),np.zeros(3)]), np.round(np.dot(orig_affine, homogeneous_coord),3)]


# determine the coordinate system of the FCSV
coord_sys,head_info=determineFCSVCoordSystem(acpc_file[0])
head_info=dict(ChainMap(*[{i:x} for i,x in enumerate(head_info)]))
fcsv_data = pd.read_csv(acpc_file[0], skiprows=3, header=None)
fcsv_data=fcsv_data.iloc[:,:].rename(columns=head_info).reset_index(drop=True)

rewrite=False
if fcsv_data.shape[1] != len(head_info):
	fcsv_data = fcsv_data[list(head_info.values())[::-1]]
	rewrite=True

if rewrite:
	with open(acpc_file[0], 'w') as fid:
		fid.write("# Markups fiducial file version = GG\n")
		fid.write(f"# CoordinateSystem = {coord_sys}\n")
		fid.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
	
	fcsv_data.round(6).to_csv(acpc_file[0], sep=',', index=False, lineterminator="", mode='a', header=False, float_format='%.6f')

coord_sys,head_info=determineFCSVCoordSystem(acpc_file[0])
head_info=dict(ChainMap(*[{i:x} for i,x in enumerate(head_info)]))
acpc_data = pd.read_csv(acpc_file[0], skiprows=3, header=None)
acpc_data=acpc_data.iloc[:,:].rename(columns=head_info).reset_index(drop=True)
if any(x in coord_sys for x in {'LPS','1'}):
	acpc_data['x'] = -1 * acpc_data['x'] # flip orientation in x
	acpc_data['y'] = -1 * acpc_data['y'] # flip orientation in y

ac_point = acpc_data.loc[acpc_data['label'] =='ac', 'x':'z'].values[0]
pc_point = acpc_data.loc[acpc_data['label'] =='pc', 'x':'z'].values[0]
mid_point = acpc_data.loc[acpc_data['label'] =='mid', 'x':'z'].values[0]
mcp = np.array([(ac_point[0] + pc_point[0]) / 2, (ac_point[1] + pc_point[1]) / 2, (ac_point[2] + pc_point[2]) / 2])*np.array([1,1,1])

riiMatrix=acpc_transform(ac_point, pc_point, mid_point)
lps2ras=np.diag([-1, -1, 1, 1])
ras2lps=np.diag([-1, -1, 1, 1])
outMatrix=centering_transform=np.dot(ras2lps,np.dot(np.linalg.inv(riiMatrix),lps2ras))
outMatrix[:3,-1]=mcp-centering_transform_raw[:3,-1]



output_matrix_txt = make_bids_filename(isub, 'T1w', None, 'mcp.tfm', patient_output)
Parameters = " ".join([str(x) for x in np.concatenate((outMatrix[0:3,0:3].reshape(9), outMatrix[:,3]))])
with open(output_matrix_txt, 'w') as fid:
	fid.write("#Insight Transform File V1.0\n")
	fid.write("#Transform 0\n")
	fid.write("Transform: AffineTransform_double_3_3\n")
	fid.write("Parameters: " + Parameters + "\n")
	fid.write("FixedParameters: 0 0 0\n")
