#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 17:45:32 2024

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
np.set_printoptions(precision=3,suppress=True)
def orient_to_ras(orig_nifti):
	x, y, z = nb.aff2axcodes(orig_nifti.affine)
	orig_aff=orig_nifti.affine.copy()
	orientation = nb.io_orientation(orig_aff)
	for k,i in enumerate(orientation[:,1]):
		if i == -1.0:
			orig_aff = np.flip(orig_aff,k)
	return orig_aff
    
    return img_trans
def check_orientation(orig_nifti):

	x, y, z = nb.aff2axcodes(orig_nifti.affine)
	orig_aff=orig_nifti.get_fdata().copy()
	
	if x != 'R':
		orig_aff = np.flip(orig_aff, 0)
	if y != 'A':
		orig_aff = np.flip(orig_aff, 1)
	if z != 'S':
		orig_aff = np.flip(orig_aff, 2)
	return orig_aff


nii_fname=r'/home/greydon/Documents/data/SEEG/bids/sub-P154/ses-pre/pet/sub-P154_ses-pre_task-rest_run-01_pet.nii.gz'
nii_fname=r'/home/greydon/Documents/data/SEEG/bids/sub-P072/ses-pre/pet/sub-P072_ses-pre_task-rest_run-01_pet.nii.gz'
nii_fname=r'/home/greydon/Documents/data/SEEG/bids/sub-P054/ses-pre/pet/sub-P054_ses-pre_task-rest_run-01_pet.nii.gz'

out_fname=os.path.join(os.path.dirname(nii_fname), os.path.basename(nii_fname).split('.nii')[0]+'__.nii.gz')


orig_nifti=nb.load(nii_fname)
if len(orig_nifti.shape)>3:
	avg = np.mean(orig_nifti.get_fdata(), axis=3)
else:
	avg=orig_nifti.get_fdata()

ornt_my = nb.orientations.io_orientation(orig_nifti.affine)
ornt_lps = nb.orientations.axcodes2ornt(('R',"A","S"))
ornt = nb.orientations.ornt_transform(ornt_my,ornt_lps)
data3d_ornt = nb.orientations.apply_orientation(avg, ornt)
t_aff = nb.orientations.inv_ornt_aff(ornt_my, data3d_ornt.shape)
out_aff = np.dot(orig_nifti.affine, t_aff)
nimg = nb.Nifti1Image(data3d_ornt.astype(np.float32), out_aff)
nimg.set_qform(nimg.affine,1)
nimg.set_sform(nimg.affine,1)
nb.save(nimg, out_fname)


vox2ras_tkr=get_vox2ras_tkr(orig_nifti)
ras2vox_tkr = np.linalg.inv(vox2ras_tkr)
vox2ras = orig_nifti.header.get_vox2ras()

nb.aff2axcodes(orig_nifti.affine)
canonical_img = nb.as_closest_canonical(orig_nifti)
canonical_img.set_sform(canonical_img.affine, code=1)
canonical_img.set_qform(canonical_img.affine, code=1)
	
nb.aff2axcodes(canonical_img.affine)
nb.save(canonical_img,out_fname)
center_coordinates=np.array([x/ 2 for x in orig_nifti.header["dim"][1:4]-1])
homogeneous_coord = np.concatenate((center_coordinates, np.array([1])), axis=0)
centering_transform_raw=np.c_[np.vstack([np.eye(3),np.zeros(3)]), np.round(np.dot(orig_affine,homogeneous_coord),3)]
print(canonical_img.header)

nb.orientations.io_orientation(canonical_img.affine)

nb.orientations.ornt2axcodes(nb.orientations.io_orientation(orig_affine))

nb.orientations.ornt2axcodes(nb.orientations.ornt_transform(nb.orientations.io_orientation(orig_nifti.affine), nb.orientations.io_orientation(canonical_img.affine)))