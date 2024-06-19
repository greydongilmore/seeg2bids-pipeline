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
import os,json
import pandas as pd
from collections import ChainMap
np.set_printoptions(precision=3,suppress=True)


def sorted_nicely(lst):
	def convert(text): return int(text) if text.isdigit() else text
	def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
	sorted_lst = sorted(lst, key=alphanum_key)
	return sorted_lst

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


def parseSTEALTHfile(istealth):
	
	with open(istealth) as (file):
		stealth_json = json.load(file)
	
	for ipoint in ('AC','PC','Midline'):
		stealth_json[ipoint]={}
		for icoordSys in ('DCM','LPS','RPI'):
			stealth_json[ipoint][icoordSys]=np.array([float(stealth_json['ACPC'][icoordSys][ipoint]['x']),float(stealth_json['ACPC'][icoordSys][ipoint]['y']),float(stealth_json['ACPC'][icoordSys][ipoint]['z'])])
	
	stealth_json['points']={}
	for icoordSys in ('DCM','LPS','RPI'):
		for idx,ipoint in enumerate(stealth_json['annotations'][icoordSys]):
			if ipoint['name'] not in list(stealth_json['points']):
				stealth_json['points'][ipoint['name']]={}
			stealth_json['points'][ipoint['name']][icoordSys]=np.array([float(ipoint['point']['x']),float(ipoint['point']['y']),float(ipoint['point']['z'])])
			
	for idx,iexam in enumerate(stealth_json['exams']):
		for itrans in ('dicomTvolMM','volumeTreference','rpiTvolMM'):
			if itrans in list(iexam):
				stealth_json['exams'][idx][itrans]=np.stack((np.array([float(iexam[itrans]['00']), float(iexam[itrans]['01']), float(iexam[itrans]['02']), float(iexam[itrans]['03'])]),
					np.array([float(iexam[itrans]['10']), float(iexam[itrans]['11']), float(iexam[itrans]['12']), float(iexam[itrans]['13'])]),
					np.array([float(iexam[itrans]['20']), float(iexam[itrans]['21']), float(iexam[itrans]['22']), float(iexam[itrans]['23'])]),
					np.array([float(iexam[itrans]['30']), float(iexam[itrans]['31']), float(iexam[itrans]['32']), float(iexam[itrans]['33'])])))
	
	for idx,iexam in enumerate(stealth_json['exams']):
		if stealth_json['exams'][idx]['designations']:
			stealth_json['dicomTvolMM']=stealth_json['exams'][idx]['dicomTvolMM']
	
	stealth_json['trajectories']={}
	for icoords in list(stealth_json['plans'].keys()):
		for iplan in stealth_json['plans'][icoords]:
			if iplan['name'] not in list(stealth_json['trajectories']):
				stealth_json['trajectories'][iplan['name']]={}
			for itype in ('entry','target'):
				if icoords not in list(stealth_json['trajectories'][iplan['name']]):
					stealth_json['trajectories'][iplan['name']][icoords]={}
				stealth_json['trajectories'][iplan['name']][icoords][itype]=np.array([float(iplan[itype]['x']), float(iplan[itype]['y']), float(iplan[itype]['z'])])
	
	return stealth_json

def writeFCSV(coords,labels,output_fcsv,coordsys='0'):
	
	with open(output_fcsv, 'w') as fid:
		fid.write("# Markups fiducial file version = 4.11\n")
		fid.write(f"# CoordinateSystem = {coordsys}\n")
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
		out_df['label'].append('')
		out_df['description'].append(ilabels)
		out_df['associatedNodeID'].append('')
	
	out_df=pd.DataFrame(out_df)
	out_df.round(3).to_csv(output_fcsv, sep=',', index=False, lineterminator="", mode='a', header=False)


#%%


stealth_file_path=r'C:\Users\greydon\Documents\data\moncton\derivatives\seeg_scenes'

isub='sub-P001'

nii_fname=glob.glob(os.path.join(stealth_file_path, isub,'*-contrast*_T1w.nii.gz'))
stealth_fname=sorted_nicely(glob.glob(os.path.join(stealth_file_path, isub,'StealthExport*.json')))


if nii_fname and stealth_fname:
	cnt=1
	for istealth in stealth_fname:
		out_fcsv=os.path.join(stealth_file_path, isub, f'{isub}_run-{str(cnt).zfill(2)}_planned.fcsv')
		out_world_fcsv=os.path.join(stealth_file_path,isub,f'{isub}_space-world_run-{str(cnt).zfill(2)}_planned.fcsv')
		out_tfm=os.path.join(stealth_file_path,isub,f'{isub}_from-subject_to-world_planned.tfm')
		out_inv_tfm=os.path.join(stealth_file_path,isub,f'{isub}_from-world_to-subject_planned.tfm')
		
		lps2ras=np.diag([-1, -1, 1, 1])
		ras2lps=np.diag([-1, -1, 1, 1])
		
		#centering transform
		orig_nifti=nb.load(nii_fname[0])
		orig_affine=orig_nifti.affine
		center_coordinates=np.array([x/ 2 for x in orig_nifti.header["dim"][1:4]-1.0])
		homogeneous_coord = np.concatenate((center_coordinates, np.array([1])), axis=0)
		centering_transform_raw=np.c_[np.r_[np.eye(3),np.zeros((1,3))], np.round(np.dot(orig_affine, homogeneous_coord),3)]
		
		#store two transforms to file, to-world and to-t1w
		for itype,ifcsv in zip(['world','t1w'],[out_inv_tfm,out_tfm]):
			if itype=='t1w':
				centering_transform=np.linalg.inv(np.dot(ras2lps,np.dot(np.linalg.inv(centering_transform_raw),lps2ras)))
				coordsys='0'
			else:
				centering_transform=np.dot(ras2lps,np.dot(np.linalg.inv(centering_transform_raw),lps2ras))
				coordsys='0'
			
			Parameters = " ".join([str(x) for x in np.concatenate((centering_transform[0:3,0:3].reshape(9), centering_transform[0:3,3]))])
			
			with open(ifcsv, 'w') as fid:
				fid.write("#Insight Transform File V1.0\n")
				fid.write("#Transform 0\n")
				fid.write("Transform: AffineTransform_double_3_3\n")
				fid.write("Parameters: " + Parameters + "\n")
				fid.write("FixedParameters: 0 0 0\n")
		
		#parse ROS file
		stealth_parsed=parseSTEALTHfile(istealth)
		
		if 'dicomTvolMM' in list(stealth_parsed):
			dcm2MM_fcsv=os.path.join(stealth_file_path, isub,f'{isub}_dicomTvolMM.tfm')
			Parameters = " ".join([str(x) for x in np.concatenate((stealth_parsed['dicomTvolMM'][0:3,0:3].reshape(9),stealth_parsed['dicomTvolMM'][0:3,3]))])
			
			with open(dcm2MM_fcsv, 'w') as fid:
				fid.write("#Insight Transform File V1.0\n")
				fid.write("#Transform 0\n")
				fid.write("Transform: AffineTransform_double_3_3\n")
				fid.write("Parameters: " + Parameters + "\n")
				fid.write("FixedParameters: 0 0 0\n")
			
		for icoordSys in ('AC','PC','MCP'):
			coords=[stealth_parsed['AC'][icoordSys],stealth_parsed['PC'][icoordSys],stealth_parsed['Midline'][icoordSys]]
			labels=['ac','pc''mid']
			ifcsv=os.path.join(stealth_file_path, isub, f'{isub}_type-{icoordSys}_acpc.fcsv')
			coordsys='0'
			writeFCSV(coords,labels,ifcsv,coordsys)
		
		(stealth_parsed['trajectories']['Plan 1']['LPS']['target']-stealth_parsed['points']['MCP']['LPS'])*np.array([-1,-1,1])
		(stealth_parsed['trajectories']['Plan 1']['LPS']['entry']-stealth_parsed['points']['MCP']['LPS'])*np.array([-1,-1,1])
		(stealth_parsed['trajectories']['Plan 2']['LPS']['target']-stealth_parsed['points']['MCP']['LPS'])*np.array([-1,-1,1])
		(stealth_parsed['trajectories']['Plan 2']['LPS']['entry']-stealth_parsed['points']['MCP']['LPS'])*np.array([-1,-1,1])
		
		for itype,ifcsv in zip(['world','t1w'],[out_world_fcsv,out_fcsv]):
			coords=[]
			labels=[]
			
			for idx,traj in enumerate(rosa_parsed['trajectories']):
				vecT = np.hstack([traj['target'],1])
				vecE = np.hstack([traj['entry'],1])
				
				if itype == 'world':
					tvecT = vecT.T
					tvecE = vecE.T
					coordsys='0'
				else:
					tvecT = centering_transform_raw @ vecT.T
					tvecE = centering_transform_raw @ vecE.T
					coordsys='0'
				
				traj['target_t']=np.round(tvecT,3).tolist()[:3]
				coords.append(traj['target_t'])
				labels.append(traj['name'])
				
				traj['entry_t']=np.round(tvecE,3).tolist()[:3]
				coords.append(traj['entry_t'])
				labels.append(traj['name'])
				
				rosa_parsed['trajectories'][idx]=traj
			
			writeFCSV(coords,labels,ifcsv,coordsys)
		
	print(f"Done {isub}")
	
