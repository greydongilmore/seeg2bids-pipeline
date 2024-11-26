#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 01:19:14 2020

@author: greydon
"""
import pandas as pd
import numpy as np
import nibabel as nib
import os,glob


debug = False

if debug:
	class dotdict(dict):
		"""dot.notation access to dictionary attributes"""
		__getattr__ = dict.get
		__setattr__ = dict.__setitem__
		__delattr__ = dict.__delitem__
	
	class Namespace:
		def __init__(self, **kwargs):
			self.__dict__.update(kwargs)
	
	isub="P002"
	data_dir=r'/home/greydon/Documents/data/emory_peds/derivatives'
	
	input=dotdict({'fcsv':f'{data_dir}/seeg_coordinates/' + f'sub-{isub}/sub-{isub}_space-native_SEEGA.fcsv',
				'fcsv_template':f'{data_dir}/seeg_coordinates/' + f'sub-{isub}/sub-{isub}_space-MNIPediatricAsymCohort4_SEEGA.fcsv',
				'dseg_tsv':'/home/greydon/Documents/GitHub/ieegProc/resources/tpl-MNI152NLin2009cSym/tpl-MNI152NLin2009cSym_atlas-CerebrA_dseg.tsv',
				'dseg_nii':f'{data_dir}/atlasreg/' + f'sub-{isub}/sub-{isub}_label-dilated_desc-nonlin_atlas-CerebrA_from-MNIPediatricAsymCohort4_dseg.nii.gz',
				'tissue_seg':f'{data_dir}/atlasreg/' + f'sub-{isub}/sub-{isub}_label-*_desc-atropos3seg_probseg.nii.gz'
				})
	
	output=dotdict({'html':f'{data_dir}/atlasreg/' + f'sub-{isub}/qc/sub-{isub}_space-MNIPediatricAsymCohort_desc-affine_electrodes.html',
				'png':f'{data_dir}/atlasreg/' + f'sub-{isub}/qc/sub-{isub}_space-MNIPediatricAsymCohort_desc-affine_electrodevis.png'
				})
	config=dotdict({'tissue_labels':['GM','WM','CSF'],
				})
	snakemake = Namespace(output=output, input=input,config=config)


#read fcsv electrodes file
df_elec = pd.read_table(snakemake.input.fcsv,sep=',',header=2)
df_elec
df_atlas = pd.read_table(snakemake.input.dseg_tsv)
df_atlas
df_template = pd.read_table(snakemake.input.fcsv_template,sep=',',header=2)
df_template

#load up tissue probability, warped from template
tissue_prob_vol = dict()
tissue_prob_elec = dict()

if isinstance(snakemake.input.tissue_seg, str):
	tissue_segs=glob.glob(snakemake.input.tissue_seg)
else:
	tissue_segs=snakemake.input.tissue_seg

for label,nii in zip(snakemake.config['tissue_labels'], tissue_segs):
	tissue_prob_vol[label] = nib.load(nii).get_fdata()
	tissue_prob_elec[label] = list()

#load dseg nii (as integer)
dseg_nii = nib.load(snakemake.input.dseg_nii)
dseg_vol = dseg_nii.get_fdata().astype('int')

#get affine from image, so we can go from RAS coords to array indices
dseg_affine = dseg_nii.affine
dseg_affine

#get coords from fcsv
coords = df_elec[['x','y','z']].to_numpy()

labelnames = []

for i in range(len(coords)):

	vec = np.hstack([coords[i,:],1])

	#dseg_affine is used to xfm indices to RAS coords, 
	# so we use the inverse to go the other way
	tvec = np.linalg.inv(dseg_affine) @ vec.T   
	inds = np.round(tvec[:3]).astype('int')

	if inds[0] < dseg_vol.shape[0] and inds[1] < dseg_vol.shape[1]:
		labelnum = dseg_vol[inds[0],inds[1],inds[2]]
	else:
		labelnum=0
	
	if labelnum >0:
		labelnames.append(df_atlas.loc[df_atlas['label']==labelnum,'name'].to_list()[0])
	else:
		labelnames.append('None')
	

	for label in snakemake.config['tissue_labels']:
		if inds[0] < dseg_vol.shape[0] and inds[1] < dseg_vol.shape[1]:
			tissue_prob_elec[label].append(tissue_prob_vol[label][inds[0],inds[1],inds[2]])
		else:
			tissue_prob_elec[label].append(0)
	
#add new columns to existing dataframe
df_elec['atlas_label'] = labelnames
for label in snakemake.config['tissue_labels']:
	df_elec[label] = tissue_prob_elec[label]

#create new dataframe with selected variables and save it
out_df = df_elec[['label','atlas_label'] + snakemake.config['tissue_labels'] + ['x','y','z']]

out_df['mni_x']=df_template['x']
out_df['mni_y']=df_template['y']
out_df['mni_z']=df_template['z']

out_df.to_csv(snakemake.output.tsv,sep='\t',float_format='%.3f',index=False)
out_df.to_excel(os.path.splitext(snakemake.output.tsv)[0]+'.xlsx',float_format='%.3f',index=False)

out_df