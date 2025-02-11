#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 22:13:00 2023

@author: greydon
"""

import numpy as np
import regex as re
import pydicom
import shutil
from functools import reduce
import nibabel as nb
import glob
import os,sys
import subprocess
import pandas as pd
from collections import ChainMap
import pandas as pd
from pathlib import Path


np.set_printoptions(precision=3,suppress=True)

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
	
	with open(ros_fname, 'r') as f:
		textfile = f.read()
	
	rosadata=getMeta(r'\[BEGIN\]\n(.+?)\n\[SERIE_UID\]', textfile)[0]
	rosadata['ATFormRAS'] = np.diag([-1,-1,1,1])
	rosadata['ac']=getCoordinates(textfile, queryPoint='AC',queryHead='ACPC').tolist()
	rosadata['pc']=getCoordinates(textfile, queryPoint='PC',queryHead='ACPC').tolist()
	rosadata['ih']=getCoordinates(textfile, queryPoint='IH',queryHead='ACPC').tolist()
	rosadata['trajectories']=getTrajectoriesList(textfile)
	rosadata['volumes']=getMeta(r'\[IMAGERY_TYPE\]\n(.+?)\n\[IMAGERY_3DREF\]', textfile)
	rosadata['robot']=getMeta(r'\[ROBOT\]\n(.+?)\n\[END\]', textfile)
	
	return rosadata

def getMeta(search_str, textfile):
	displays=[]
	result = re.findall(search_str, textfile, re.DOTALL)
	for iresult in result:
		data = dict(ChainMap(*extractTokens(iresult)))
		for key,value in data.items():
			temp=value.split('\n')[-1].strip()
			if key.startswith('TR'):
				temp=np.array([float(x) for x in temp.split(' ')])
				if temp.shape[0]>1:
					temp=temp.reshape(4, 4)
			data[key]=temp
		displays.append(data)
	
	return displays

def getCoordinates(textfile, queryPoint, queryHead):
	pattern = rf"(?<=\[{queryHead}\]).*" + queryPoint + r" \d -?\d+\.\d+ -?\d+\.\d+ -?\d+\.\d+"
	m = re.search(pattern, textfile, re.DOTALL)
	coords_str = m.group().split(' ')[-3:]
	coords_lps = np.array(list(map(float, coords_str)))
	return coords_lps

def getTrajectoriesList(textfile):
	pattern = r"(?P<name>[\^\-+\w]+) (?P<type>\d) (?P<color>\d+) (?P<entry_point_defined>\d) (?P<entry>-?\d+\.\d+ -?\d+\.\d+ -?\d+\.\d+) (?P<target_point_defined>\d) (?P<target>-?\d+\.\d+ -?\d+\.\d+ -?\d+\.\d+) (?P<instrument_length>\d+\.\d+) (?P<instrument_diameter>\d+\.\d+)\n"
	trajectories = [m.groupdict() for m in re.finditer(pattern, textfile)]
	for trajectory in trajectories:
		trajectory['name']=trajectory['name'].replace('^',' ')
		for pos in ['entry', 'target']:
			trajectory[pos] = np.array(list(map(float, trajectory[pos].split(' ')))) # str to array
	return trajectories

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

def spm_affine(in_file):
	"""Returns the affine transform of a nifti image.Mimics the spm function spm_get_space.
	Parameters
	----------
	in_file : str
		Path to an existant nifti image.
	Returns
	-------
	affine : numpy.ndarray of shape (4, 4)
		The affine transformation matrix.
	"""
	img = nb.load(in_file)
	affine = img.affine.copy()
	rotation = affine[:3, :3]
	chol = np.linalg.cholesky(rotation.T.dot(rotation))
	zooms = np.diag(chol).copy()
	if np.linalg.det(rotation) < 0:
		zooms[0] *= -1
	affine[:3, 3] = affine[:3, 3] - zooms * np.ones((3, ))
	return affine

def rotation_matrix(alpha_x, alpha_y, alpha_z):
	#pitch, roll, yaw = np.array([pitch, roll, yaw]) * np.pi / 180
	rotate_x = np.array([
		[1, 0, 0, 0],
		[0, np.cos(alpha_x), -np.sin(alpha_x), 0],
		[0, np.sin(alpha_x), np.cos(alpha_x), 0],
		[0, 0, 0,1]
	])
	rotate_y = np.array([
		[np.cos(alpha_y), 0, np.sin(alpha_y), 0],
		[0, 1, 0, 0],
		[-np.sin(alpha_y), 0, np.cos(alpha_y), 0],
		[0, 0, 0,1]
	])
	rotate_z = np.array([
		[np.cos(alpha_z), -np.sin(alpha_z), 0, 0],
		[np.sin(alpha_z), np.cos(alpha_z), 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0,1]
	])
	return rotate_z @ rotate_y @ rotate_x

def rotation3d(x=0, y=0, z=0):
	cos_x = np.cos(x);cos_y = np.cos(y);cos_z = np.cos(z)
	sin_x = np.sin(x);sin_y = np.sin(y);sin_z = np.sin(z)
	r = np.array(
		[
			[cos_y * cos_z, -cos_x * sin_z + sin_x * sin_y * cos_z, sin_x * sin_z + cos_x * sin_y * cos_z, 0],
			[cos_y * sin_z, cos_x * cos_z + sin_x * sin_y * sin_z, -sin_x * cos_z + cos_x * sin_y * sin_z, 0],
			[-sin_y, sin_x * cos_y, cos_x * cos_y, 0],
			[0, 0, 0,1]
		],dtype=float,
	)
	return r

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

def getFrameRotation(ac,pc,mid):
	pmprime = (ac + pc) / 2
	vec1 = ac - pmprime
	vec2 = mid - pmprime
	vec1Mag = np.sqrt(vec1[0] ** 2 + vec1[1] ** 2 + vec1[2] ** 2)
	vec2Mag = np.sqrt(vec2[0] ** 2 + vec2[1] ** 2 + vec2[2] ** 2)
	vec1Unit = vec1 / vec1Mag
	vec2Unit = vec2 / vec2Mag
	yihatprime = vec1Unit
	if pmprime[2] > mid[2]:
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

def xfm_txt_to_tfm(xfm_fname):
	transformMatrix = np.loadtxt(xfm_fname)
	lps2ras=np.diag([-1, -1, 1, 1])
	ras2lps=np.diag([-1, -1, 1, 1])
	transform_lps=np.dot(ras2lps, np.dot(transformMatrix,lps2ras))
	
	Parameters = " ".join([str(x) for x in np.concatenate((transform_lps[0:3,0:3].reshape(9), transform_lps[0:3,3]))])
	
	tfm_fname=os.path.splitext(xfm_fname)[0] + '.tfm'
	with open(tfm_fname, 'w') as fid:
		fid.write("#Insight Transform File V1.0\n")
		fid.write("#Transform 0\n")
		fid.write("Transform: AffineTransform_double_3_3\n")
		fid.write("Parameters: " + Parameters + "\n")
		fid.write("FixedParameters: 0 0 0\n")
		
def run_command(cmdLineArguments, regEnv=None):
	if regEnv is not None:
		subprocess.run(cmdLineArguments, stdout=subprocess.PIPE,stderr=subprocess.STDOUT, shell=True, env=(regEnv))
	else:
		subprocess.run(cmdLineArguments, stdout=subprocess.PIPE,stderr=subprocess.STDOUT, shell=True)

def sorted_nicely(lst):
	convert = lambda text: int(text) if text.isdigit() else text
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	sorted_lst = sorted(lst, key = alphanum_key)
	
	return sorted_lst

greedy_bin=r'/opt/greedy-1.3.0/bin/greedy'

#%%

ros_file_path=r'E:\rosa'
isubpath=os.path.sep.join([ros_file_path,'uploaded'])


ros_file_path=r'/home/greydon/Documents/data/emory_seeg/derivatives/slicer_scene'
isubpath=os.path.sep.join([ros_file_path,'sub-EMOP0367'])


if glob.glob(os.path.join(ros_file_path,"sub-*")):
	key_word='sub-*'
else:
	key_word='*'

for isubpath in sorted_nicely(glob.glob(os.path.join(ros_file_path,key_word))):
	isub=os.path.basename(isubpath)
	ros_fname_list=glob.glob(os.path.join(ros_file_path, isub,"**",'*.ros'),recursive=True)
	for ros_fname in sorted_nicely(ros_fname_list)[::-1]:
		#parse ROS file
		rosa_parsed=parseROSAfile(ros_fname)
		
		dicom_dir_list=glob.glob(os.path.join(ros_file_path, isub,"**",'DICOM'),recursive=True)
		
		for dicom_dir in dicom_dir_list:
			for idicom in sorted_nicely(os.listdir(dicom_dir)):
				dicom_files=[]
				for root, directories, filenames in os.walk(os.path.join(dicom_dir,idicom)):
					for filename in filenames:
						full_filename=os.path.join(root, filename)
						if pydicom.misc.is_dicom(full_filename):
							dicom_files.append(full_filename)
				
				if dicom_files:
					out_dir=os.path.join(dicom_dir,idicom,'DICOMFiles')
					if not os.path.exists(f'{out_dir}.zip'):
						if not os.path.exists(out_dir):
							os.makedirs(out_dir, exist_ok=True)
							folder=Path(out_dir)
							for ifile in dicom_files:
								shutil.move(ifile, folder)
						
						if sys.platform == 'win32':
							zip_cmd = ' '.join([f'{os.path.splitroot(out_dir)[0].lower()}&&',
												  f'cd {os.path.dirname(os.path.splitroot(out_dir)[-1])}&&',
												 'zip -r',
												 f'"{out_dir}.zip"',
												 f'{os.path.basename(out_dir)}'])
						else:
							zip_cmd = ' '.join([f'cd {out_dir}&&',
												 'zip -2 -r -m',
												 f'"{out_dir}.zip"',
												 '*'])
						
						print(f"Zipping {os.path.basename(os.path.dirname(dicom_dir))}: {idicom}")
						run_command(zip_cmd)
						shutil.rmtree(out_dir)
					else:
						print(f"EXISTS: {out_dir}.zip")
		
		
		nii_xfm=glob.glob(os.path.join(isubpath, f"{isub}_acq-ROSA_desc-rigid_*_xfm.tfm"))
		sub2t_xfm=None
		if nii_xfm:
			nii_xfm=nii_xfm[0]
			nii_fname=glob.glob(os.path.join(isubpath, f"{isub}_ses-pre_acq-ROSA_*.nii.gz"))
			
			sub2template= pd.read_table(nii_xfm,sep=",",header=2)
			sub2t_xfm=np.array([float(x) for x in sub2template.iloc[0,0].split(':')[-1].strip().split(' ')])
			sub2t_transform = np.eye(4)
			sub2t_transform[0:3, 0:3] = sub2t_xfm[:9].reshape(3, 3)
			sub2t_transform[0:3, 3] = sub2t_xfm[9:]
			
			lps2ras=np.diag([-1, -1, 1, 1])
			ras2lps=np.diag([-1, -1, 1, 1])
			sub2t_xfm=np.dot(ras2lps,np.dot(np.linalg.inv(sub2t_transform),lps2ras))
		else:
			nii_fname=glob.glob(f"{ros_file_path}/{isub}/*-contrast*_T1w.nii.gz")
			
			
		
		rot2ras=rotation_matrix(np.deg2rad(0),np.deg2rad(0),np.deg2rad(180))
		acpc_fname=glob.glob(os.path.join(ros_file_path, isub,'*acpc.fcsv'))
		out_tfm=os.path.join(ros_file_path,isub,f'{isub}_from-subject_to-world_planned.tfm')
		out_acpc_tfm=os.path.join(ros_file_path,isub,f'{isub}_acpc_xfm.tfm')
		out_inv_tfm=os.path.join(ros_file_path,isub,f'{isub}_from-world_to-subject_planned.tfm')
		out_dir=os.path.join(ros_file_path, isub,'RAS_data_python')
		out_fcsv=os.path.join(ros_file_path,isub,f'{isub}_planned.fcsv')
		out_acpc_fcsv=os.path.join(ros_file_path,isub,f'{isub}_acpc.fcsv')
		
		if nii_fname:
			
			#centering transform
			orig_nifti=nb.load(nii_fname[0])
			orig_affine=orig_nifti.affine
			center_coordinates=np.array([x/ 2 for x in orig_nifti.header["dim"][1:4]-1.0])
			homogeneous_coord = np.concatenate((center_coordinates, np.array([1])), axis=0)
			centering_transform_raw=np.c_[np.r_[np.eye(3),np.zeros((1,3))], np.round(np.dot(orig_affine, homogeneous_coord),3)]
			centering_transform=np.dot(ras2lps,np.dot(np.linalg.inv(centering_transform_raw),lps2ras))
			
			Parameters = " ".join([str(x) for x in np.concatenate((centering_transform[0:3,0:3].reshape(9), centering_transform[0:3,3]))])
			with open(out_tfm, 'w') as fid:
				fid.write("#Insight Transform File V1.0\n")
				fid.write("#Transform 0\n")
				fid.write("Transform: AffineTransform_double_3_3\n")
				fid.write("Parameters: " + Parameters + "\n")
				fid.write("FixedParameters: 0 0 0\n")
			
			coords=[]
			descs=[]
			for idx,traj in enumerate(rosa_parsed['trajectories']):
				vecT = np.hstack([traj['target'],1])*np.array([-1,-1,1,1])
				vecE = np.hstack([traj['entry'],1])*np.array([-1,-1,1,1])
				
				tvecT = centering_transform_raw@vecT
				tvecE = centering_transform_raw@vecE
				
				if sub2t_xfm is not None:
					tvecT = sub2t_xfm@tvecT
					tvecE = sub2t_xfm@tvecE
				
				coordsys='0'
				
				traj['target_t']=np.round(tvecT,3).tolist()[:3]
				coords.append(traj['target_t'])
				descs.append(traj['name'])
				
				traj['entry_t']=np.round(tvecE,3).tolist()[:3]
				coords.append(traj['entry_t'])
				descs.append(traj['name'])
				
				rosa_parsed['trajectories'][idx]=traj
			
			writeFCSV(coords,[],descs,output_fcsv=out_fcsv,coordsys=coordsys)
			
			if rosa_parsed['ac'] and rosa_parsed['pc']:
				acT = centering_transform_raw @ (np.hstack([rosa_parsed['ac'],1])*np.array([-1,-1,1,1]))
				pcT = centering_transform_raw @ (np.hstack([rosa_parsed['pc'],1])*np.array([-1,-1,1,1]))
				ihT = centering_transform_raw @ (np.hstack([rosa_parsed['ih'],1])*np.array([-1,-1,1,1]))
				
				acT = sub2t_transform @ acT
				pcT = sub2t_transform @ pcT
				ihT = sub2t_transform @ ihT
				
				coords=[acT,pcT,ihT]
				descs=['ac','pc','mid']
				
				writeFCSV(coords,descs,[],output_fcsv=out_acpc_fcsv,coordsys='0')
			
			if acpc_fname:
				coord_sys,head_info=determineFCSVCoordSystem(acpc_fname[0])
				head_info=dict(ChainMap(*[{i:x} for i,x in enumerate(head_info)]))
				fcsv_data = pd.read_csv(acpc_fname[0], skiprows=3, header=None)
				fcsv_data=fcsv_data.iloc[:,:].rename(columns=head_info).reset_index(drop=True)
				if any(x in coord_sys for x in {'LPS','1'}):
					fcsv_data['x'] = -1 * fcsv_data['x'] # flip orientation in x
					fcsv_data['y'] = -1 * fcsv_data['y'] # flip orientation in y
				
				ac_p=fcsv_data.loc[fcsv_data['label'] =='ac', 'x':'z'].values[0]
				pc_p=fcsv_data.loc[fcsv_data['label'] =='pc', 'x':'z'].values[0]
				mid_p=fcsv_data.loc[fcsv_data['label'] =='mid', 'x':'z'].values[0]
				mcp = np.array([(ac_p[0] + pc_p[0]) / 2, (ac_p[1] + pc_p[1]) / 2, (ac_p[2] + pc_p[2]) / 2])*np.array([1,1,1])
				
				riiMatrix=acpc_transform(ac_p, pc_p, mid_p)
				riiMatrix=getFrameRotation(ac_p, pc_p, mid_p)
				lps2ras=np.diag([-1, -1, 1, 1])
				ras2lps=np.diag([-1, -1, 1, 1])
				outMatrix=np.dot(ras2lps,np.dot(np.linalg.inv(riiMatrix),lps2ras))
				riiMatrix[:3,-1]=centering_transform[:3,-1]
				outACPC = " ".join([str(x) for x in np.concatenate((outMatrix[0:3,0:3].reshape(9), outMatrix[0:3,3]))])
				
				with open(out_acpc_tfm, 'w') as fid:
					fid.write("#Insight Transform File V1.0\n")
					fid.write("#Transform 0\n")
					fid.write("Transform: AffineTransform_double_3_3\n")
					fid.write("Parameters: " + outACPC + "\n")
					fid.write("FixedParameters: 0 0 0\n")
			
			
			
			print(f"Done {isub}")
			
