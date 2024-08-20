#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 21:07:53 2024

@author: greydon
"""
import os,glob,shutil,gzip
from nilearn import plotting, surface
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.collections import PolyCollection
import subprocess


regEnv = os.environ.copy()
regEnv['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
regEnv["FREESURFER_HOME"] = r'/usr/local/freesurfer/7.4.1'
regEnv["SUBJECTS_DIR"] = r'/home/greydon/Documents/data/SEEG/derivatives/fastsurfer'
regEnv["PATH"] = os.environ["PATH"]+":"+regEnv['FREESURFER_HOME']+"/bin"

def run_command2(cmdLineArguments):
	process = subprocess.Popen(cmdLineArguments, stdout=subprocess.PIPE, shell=True, stderr=subprocess.PIPE, env=regEnv)
	rc = process.poll()
	stdout = process.communicate()[0]
	return rc

def environment(sh_file=None, env=regEnv):
	command = ["bash", "-c", ". f'{sh_file}' ; /usr/bin/printenv"]
	process = subprocess.Popen(command, env=env,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	stdout, stderr = process.communicate()
	environment = {}
	for line in stdout.decode().split('\n'):
		if line.startswith("export"):
			line = line.replace("export ", b"")
			line = line.replace("'", b"")
		match = re.match(r"^(\w+)=(\S*)$", line)
		if match:
			name, value = match.groups()
			if name != "PWD":
				environment[name] = value
	return environment

def run_command(cmdLineArguments, regEnv=regEnv):
	if regEnv is not None:
		subprocess.run(cmdLineArguments, stdout=subprocess.PIPE,stderr=subprocess.STDOUT, shell=True, env=(regEnv))
	else:
		subprocess.run(cmdLineArguments, stdout=subprocess.PIPE,stderr=subprocess.STDOUT, shell=True)

fs_dir=r'/usr/local/freesurfer/7.4.1'
vol2surf_exe=os.path.join(fs_dir,'bin','mri_vol2surf')
tkregister_exe=os.path.join(fs_dir,'bin','tkregister2')
fs_shell_exe=os.path.join(fs_dir,'SetUpFreeSurfer.sh')

sh_file=fs_shell_exe
#%%


regEnv = environment(fs_shell_exe, regEnv)

data_dir=r'/home/greydon/Documents/data/SEEG'

isub='sub-P154'

fs_subdir=os.path.join(data_dir, 'derivatives', 'fastsurfer')

lh_pial_file = os.path.join(data_dir, 'derivatives', 'fastsurfer',isub, 'surf', 'lh.pial')
rh_pial_file = os.path.join(data_dir, 'derivatives', 'fastsurfer',isub, 'surf', 'rh.pial')
lh_infl_file = os.path.join(data_dir, 'derivatives', 'fastsurfer',isub, 'surf', 'lh.inflated')
rh_infl_file = os.path.join(data_dir, 'derivatives', 'fastsurfer',isub, 'surf', 'rh.inflated')
rawfile = os.path.join(data_dir, 'derivatives', 'fastsurfer', isub, 'mri', "rawavg.mgz")
origfile = os.path.join(data_dir, 'derivatives', 'fastsurfer', isub, 'mri', "orig.mgz")
petorig = glob.glob(os.path.join(data_dir, 'derivatives', 'atlasreg', isub, "*_desc-rigid_pet.nii.gz"))[0]
pet_out = os.path.join(data_dir, 'derivatives', 'fastsurfer', isub, 'convert', "pet.mgz")

petorig_img=nb.load(petorig)
nb.save(petorig_img, pet_out)


lh_data = nb.freesurfer.read_geometry(lh_pial_file)
rh_data = nb.freesurfer.read_geometry(rh_pial_file)
lh_infl_data = nb.freesurfer.read_geometry(lh_infl_file)
rh_infl_data = nb.freesurfer.read_geometry(rh_infl_file)


convertdir = os.path.join(data_dir, 'derivatives', 'fastsurfer', isub, "convert")
if not os.path.isdir(convertdir):
	os.makedirs(convertdir)


# Construct the FreeSurfer command
trffile = os.path.join(convertdir, "register.native.dat")

tkregister_cmd = ' '.join([
						   f'{tkregister_exe}',
						 '--mov',f'"{pet_out}"',
						 '--targ',f'"{origfile}"',
						 '--s', f'{isub}',
						 '--reg',f'"{trffile}"',
						 '--noedit --regheader'])

run_command(tkregister_cmd,regEnv)

cmd = ["mri_vol2surf", "--src", volume_file, "--out", out_texture_file,
           "--srcreg", dat_file, "--hemi", hemi, "--trgsubject","ico", "--icoorder", f"{ico_order}", "--surf",surface_name, "--sd", fsdir, "--srcsubject", sid, "--noreshape",
           ]

hemis = ['lh', 'rh']
for hemi in hemis:
	outfile = os.path.join(os.path.dirname(pet_out), hemi+'_pet.mgh')
	vol2surf_cmd = ' '.join([
						   f'{vol2surf_exe}',
						 '--src', f'"{pet_out}"',
						 '--reg', f'"{trffile}"',
						 '--regheader', f'{isub}',
						 "--hemi", hemi,
						 '--interp trilinear',
						 "--sd", fs_subdir,
						 "--out", f"{outfile}"
						 ])
	run_command(vol2surf_cmd,regEnv)

hemis = ['lh', 'rh']
for hemi in hemis:
	outfile = os.path.join(os.path.dirname(pet_out), hemi+'_pet.mgh')
	scalar_data = np.squeeze(nb.load(outfile).get_fdata())
	scalar_data = np.ravel(scalar_data, order="F")

