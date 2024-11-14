import numpy as np
import nibabel as nib
import json
import os

#load up tissue probability, warped from template
tissue_prob_vol = dict()

for nii in snakemake.input.tissue_priors:
	label=[x for x in os.path.basename(nii).split('_') if 'label' in x][0].split('-')[-1]
	tissue_prob_vol[label] = nib.load(nii).get_fdata()
	

#load up k-class tissue segmentation
tissue_k_seg = nib.load(snakemake.input.seg_channels_4d)
tissue_k_seg.shape

sim_prior_k = np.zeros([len(snakemake.config['tissue_labels']),tissue_k_seg.shape[3]])

#for each prior, need to find the channel that best fits
for i,label in enumerate(snakemake.config['tissue_labels']):
	for k in range(tissue_k_seg.shape[3]):

		print(f'Computing overlap of {label} prior and channel {k}... ')
		#compute intersection over union
		s1 = tissue_prob_vol[label] >0.5
		s2 = tissue_k_seg.slicer[:,:,:,k].get_fdata() >0.5
		sim_prior_k[i,k] = np.sum(np.logical_and(s1,s2).flat) / np.sum(np.logical_or(s1,s2).flat) 

label_to_k_dict = dict()

for i,label in enumerate(snakemake.config['tissue_labels']):
	label_to_k_dict[label] = int(np.argmax(sim_prior_k[i,:]))
	#write nii to file
	print('writing image at channel {} to output file: {}'.format(label_to_k_dict[label], \
													snakemake.output.tissue_segs[i]))
	nib.save(tissue_k_seg.slicer[:,:,:,label_to_k_dict[label]],\
					snakemake.output.tissue_segs[i])


with open(snakemake.output.mapping_json, 'w') as outfile:
	json.dump(label_to_k_dict, outfile,indent=4)


