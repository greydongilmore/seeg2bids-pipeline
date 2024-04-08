#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 00:43:30 2020

@author: ggilmore
"""
import os
import pandas as pd
import numpy as np
import re
import csv
from bids.layout import BIDSLayout
import shutil
from collections import OrderedDict

chan_label_dic = {
					'LAntSSMA': 'LASSMA',
					'RAntSSMA': 'RASSMA',
					'LPostSSMA': 'LPSSMA',
					'RPostSSMA': 'RPSSMA',
					'LMidIn': 'LMIn',
					'RMidIn': 'RMIn',
					'LOFr': 'LOF',
					'ROFr': 'ROF',
					'LAntOF': 'LAOF',
					'RAntOF': 'RAOF',
					'LPostOF': 'LPOF',
					'RPostOF': 'RPOF',
					'LFACing': 'LACg',
					'RFACing': 'RACg',
					'LAMesFr': 'LAMeFr',
					'RAMesFr': 'RAMeFr',
					'LPMesFr': 'LPMeFr',
					'RPMesFr': 'RPMeFr',
					'LAmy': 'LAm',
					'RAmy': 'RAm',
					'LTAmy': 'LAm',
					'RTAmy': 'RAm',
					'LTPole': 'LTeP',
					'RTPole': 'RTeP',
					'LTAHc': 'LAHc',
					'RTAHc': 'RAHc',
					'LTPHc': 'LPHc',
					'RTPHc': 'RPHc',
					'LPost_Central': 'LPCe',
					'RPost_Central': 'RPCe',
					'LFr_Convex': 'LFrC',
					'RFr_Convex': 'RFrC',
					'LPost_to_Les': 'LPLs',
					'RPost_to_Les': 'RPLs',
					'LFAnt_to_Les': 'LALs',
					'RFAnt_to_Les': 'RALs',
					'LTPO_PostCing': 'LPCg',
					'RTPO_PostCing': 'RPCg',
					'LTPOPostCing': 'LPCg',
					'RTPOPostCing': 'RPCg',
					'LAntCing': 'LACg',
					'RAntCing': 'RACg',
					'LPostTe_MedOcc': 'LPTMOc',
					'RPostTe_MedOcc': 'RPTMOc',
					'LPreCent_Face': 'LPrCeP',
					'RPreCent_Face': 'RPrCeP',
					'LFACing': 'LACg',
					'RFACing': 'RACg',
					'LSensory_Cx_Leg': 'LSleg',
					'RSensory_Cx_Leg': 'RSleg',
					'LMotor_Cx_Leg': 'LMleg',
					'RMotor_Cx_Leg': 'RMleg',
					'LLesion_SUP': 'LSLs',
					'RLesion_SUP': 'RSLs',
					'LLesion_POST': 'LPLs',
					'RLesion_POST': 'RPLs',
					'LLesion_INF': 'LILs',
					'RLesion_INF': 'RILs',
					'LLesion_ANT': 'LALs',
					'RLesion_ANT': 'RALs',
					'LTFusiform': 'LFGy',
					'RTFusiform': 'RFGy',
					'LFOF': 'LOFr',
					'RFOF': 'LOFr',
					'LSupMargPCing': 'LPCg',
					'RSupMargPCing': 'RPCg',
					'LTHeschl': 'LHs',
					'RTHeschl': 'RHs',
					}

def determine_groups(iterable, numbered_labels=False):
	values = []
	for item in iterable:
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
	
	return values_unique,count


def levenshtein_ratio_and_distance(s, t, ratio_calc = False):
	""" levenshtein_ratio_and_distance:
		Calculates levenshtein distance between two strings.
		If ratio_calc = True, the function computes the
		levenshtein distance ratio of similarity between two strings
		For all i and j, distance[i,j] will contain the Levenshtein
		distance between the first i characters of s and the
		first j characters of t
	"""
	# Initialize matrix of zeros
	rows = len(s)+1
	cols = len(t)+1
	distance = np.zeros((rows,cols),dtype = int)

	# Populate matrix of zeros with the indeces of each character of both strings
	for i in range(1, rows):
		for k in range(1,cols):
			distance[i][0] = i
			distance[0][k] = k

	# Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
	for col in range(1, cols):
		for row in range(1, rows):
			if s[row-1] == t[col-1]:
				cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
			else:
				# In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
				# the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
				if ratio_calc == True:
					cost = 2
				else:
					cost = 1
			distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
								 distance[row][col-1] + 1,          # Cost of insertions
								 distance[row-1][col-1] + cost)     # Cost of substitutions
	if ratio_calc == True:
		# Computation of the Levenshtein Distance Ratio
		Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
		return Ratio
	else:
		# print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
		# insertions and/or substitutions
		# This is the minimum number of edits needed to convert string a to string b
		return "The strings are {} edits away".format(distance[row][col])

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
		if any(headFlag.match(x) for x in list(cleaned_dict)):
			headFin=list(cleaned_dict.values())[0].split(',')
	
	if any(x in coord_sys for x in {'LPS','1'}):
		df = pd.read_csv(input_fcsv, skiprows=3, header=None)
		df[1] = -1 * df[1] # flip orientation in x
		df[2] = -1 * df[2] # flip orientation in y
		
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
	
	sub='P009'
	
	config=dotdict({'out_dir':'/home/greydon/Documents/data/SEEG_peds'})
	#config=dotdict({'out_dir':'/media/stereotaxy/3E7CE0407CDFF11F/data/SEEG/imaging/clinical'})
	
	params=dotdict({'sub':sub})
	input=dotdict({'seega_scene':f'/home/greydon/Documents/data/SEEG_peds/derivatives/seega_scenes/sub-{sub}'})
	#input=dotdict({'seega_scene':f'/home/greydon/Documents/data/SEEG/derivatives/seega_scenes/sub-{sub}'})
	
	snakemake = Namespace(params=params, input=input,config=config)

#%%

isub='sub-'+snakemake.params.sub

patient_output = os.path.dirname(snakemake.output.seega_fcsv)
#if not os.path.exists(patient_output):
#	os.makedirs(patient_output)

patient_files = []
for dirpath, subdirs, subfiles in os.walk(os.path.dirname(snakemake.input.seega_scene)):
	for x in subfiles:
		if x.endswith(".fcsv") and not x.startswith('coords'):
			patient_files.append(os.path.join(dirpath, x))


acpc_file = [x for x in patient_files if os.path.splitext(x)[0].lower().endswith('acpc')]
patient_files = [x for x in patient_files if any(os.path.splitext(x)[0].lower().endswith(y) for y in ('seega','seeg','planned','actual'))]

if acpc_file:
	
	# determine the coordinate system of the FCSV
	determineFCSVCoordSystem(acpc_file[0])
	
	acpc_data = pd.read_csv(acpc_file[0], skiprows=3, header=None)
	acpc_data.rename(columns={0:'node_id', 1:'x', 2:'y', 3:'z', 4:'ow', 5:'ox',
						6:'oy', 7:'oz', 8:'vis', 9:'sel', 10:'lock',
						11:'label', 12:'description', 13:'associatedNodeID'}, inplace=True)
	ac_point = acpc_data.loc[acpc_data['label'] =='ac', 'x':'z'].values[0]
	pc_point = acpc_data.loc[acpc_data['label'] =='pc', 'x':'z'].values[0]
	mcp_point = [(ac_point[0]+pc_point[0])/2, (ac_point[1]+pc_point[1])/2, (ac_point[2]+pc_point[2])/2]
	output_matrix_txt = make_bids_filename(isub, 'T1w', None, 'mcp.tfm', patient_output)
	with open(output_matrix_txt, 'w') as fid:
		fid.write("#Insight Transform File V1.0\n")
		fid.write("#Transform 0\n")
		fid.write("Transform: AffineTransform_double_3_3\n")
		fid.write("Parameters: 1 0 0 0 1 0 0 0 1 {} {} {}\n".format(1*(round(mcp_point[0],3)), 1*(round(mcp_point[1],3)), -1*(round(mcp_point[2],3))))
		fid.write("FixedParameters: 0 0 0\n")

for ifile in patient_files:

	# determine the coordinate system of the FCSV
	determineFCSVCoordSystem(ifile)

	data_table_full = pd.read_csv(ifile, skiprows=3, header=None)
	data_table_full.rename(columns={0:'node_id', 1:'x', 2:'y', 3:'z', 4:'ow', 5:'ox',
					6:'oy', 7:'oz', 8:'vis', 9:'sel', 10:'lock',
					11:'label', 12:'description', 13:'associatedNodeID'}, inplace=True)
	
	coords_type=os.path.splitext(os.path.basename(ifile))[0].split('_')[-1]
	data_table_full['type'] = np.repeat(coords_type, data_table_full.shape[0])
	data_table_full['label']=[x.strip() for x in data_table_full['label']]

	if coords_type.lower().endswith('seega'):
		groups,n_members = determine_groups(np.array(data_table_full['label'].values), True)
		
		group_pair = []
		new_label = []
		new_group = []
		for ichan in data_table_full['label'].values:
			group_pair.append([x for x in groups if ichan.startswith(x)][0])
			if '_' in group_pair[-1]:
				group_pair[-1] = "_".join(["".join(x for x in group_pair[-1].split('_')[0] if not x.isdigit())] + group_pair[-1].split('_')[1:])
			
			if "".join(x for x in group_pair[-1] if not x.isdigit()) in list(chan_label_dic):
				temp = "".join(x for x in group_pair[-1] if not x.isdigit())
				new_group.append(chan_label_dic[temp])
				new_label.append(ichan.replace(temp, chan_label_dic[temp]))
			else:
				new_group.append("".join(x for x in group_pair[-1]))
				new_label.append(new_group[-1] + ichan.split(group_pair[-1])[-1])
		
		data_table_full.insert(data_table_full.shape[1],'orig_group',group_pair)
		data_table_full.insert(data_table_full.shape[1],'new_label',new_label)
		data_table_full.insert(data_table_full.shape[1],'new_group',new_group)
	
	if acpc_file:
		data_table_full['x_mcp'] = data_table_full['x'] - mcp_point[0]
		data_table_full['y_mcp'] = data_table_full['y'] - mcp_point[1]
		data_table_full['z_mcp'] = data_table_full['z'] - mcp_point[2]
		
		#### Write MCP Based Coords TSV file
		output_fname = make_bids_filename(isub, 'acpc', None, coords_type + '.tsv', patient_output)
		if 'seeg' in coords_type.lower():
			head = ['type','label','x_mcp','y_mcp','z_mcp','orig_group','new_label','new_group']
		else:
			head = ['type','label','x_mcp','y_mcp','z_mcp']
			
		data_table_full.round(6).to_csv(output_fname, sep='\t', index=False, na_rep='n/a', lineterminator="", columns = head, float_format='%.6f')
		
		#### Write MCP Based Coords FCSV file
		output_fname = make_bids_filename(isub, 'acpc', None, coords_type + '.fcsv', patient_output)
		with open(output_fname, 'w') as fid:
			fid.write("# Markups fiducial file version = 4.11\n")
			fid.write("# CoordinateSystem = 0\n")
			fid.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
	
		head = ['node_id', 'x_mcp', 'y_mcp', 'z_mcp', 'ow', 'ox', 'oy', 'oz', 'vis','sel', 'lock', 'label', 'description', 'associatedNodeID']
		data_table_full['node_id'] = ['vtkMRMLMarkupsFiducialNode_' + str(x) for x in range(data_table_full.shape[0])]
		data_table_full['associatedNodeID'] = np.repeat('',data_table_full.shape[0])
		data_table_full.round(6).to_csv(output_fname, sep=',', index=False, lineterminator="", columns = head, mode='a', header=False, float_format='%.6f')
	
	#### Write Native Coords TSV file
	output_fname = make_bids_filename(isub, 'native', None, coords_type + '.tsv', patient_output)
	if coords_type.lower().endswith('seega'):
		head=['type','label','x','y','z','orig_group','new_label','new_group']
	else:
		head=['type','label','x','y','z']
		
	data_table_full.round(6).to_csv(output_fname, sep='\t', index=False, na_rep='n/a', lineterminator="", columns = head, float_format='%.6f')
	
	#### Write Native Coords FCSV file
	output_fname = make_bids_filename(isub, 'native', None, coords_type + '.fcsv', patient_output)
	with open(output_fname, 'w') as fid:
		fid.write("# Markups fiducial file version = 4.11\n")
		fid.write("# CoordinateSystem = 0\n")
		fid.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
	
	head = ['node_id', 'x', 'y', 'z', 'ow', 'ox', 'oy', 'oz', 'vis','sel', 'lock', 'label', 'description', 'associatedNodeID']
	del data_table_full['node_id']
	del data_table_full['associatedNodeID']
	data_table_full.insert(data_table_full.shape[1],'node_id',pd.Series(['vtkMRMLMarkupsFiducialNode_' + str(x) for x in range(data_table_full.shape[0])]))
	data_table_full.insert(data_table_full.shape[1],'associatedNodeID', pd.Series(np.repeat('',data_table_full.shape[0])))
	data_table_full.round(6).to_csv(output_fname, sep=',', index=False, lineterminator="", columns = head, mode='a', header=False, float_format='%.6f')


coords_fname = make_bids_filename(isub, 'native', None, 'SEEGA.tsv', patient_output)
coords_table = pd.read_csv(coords_fname, sep='\t', header=0)

indexes = np.unique(coords_table['new_group'], return_index=True)[1]
slicer_chans_groups = [coords_table['new_group'][index] for index in sorted(indexes)]

indexes = np.unique(coords_table['orig_group'], return_index=True)[1]
slicer_chans_groups_orig = [coords_table['orig_group'][index] for index in sorted(indexes)]

coords_pairs = {}
coords_pairs['ieeg_labels'] = list(np.repeat(np.nan,len(slicer_chans_groups)))
coords_pairs['combined_labels'] = slicer_chans_groups
coords_pairs['seega_labels'] = slicer_chans_groups_orig

coords_pairs = pd.DataFrame(coords_pairs)
output_fname = make_bids_filename(isub, None, None, 'mapping.tsv', patient_output)
coords_pairs.to_csv(output_fname, sep='\t', index=False, na_rep='n/a', lineterminator="", float_format='%.6f')





