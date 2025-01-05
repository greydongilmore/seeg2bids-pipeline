#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:31:40 2024

@author: greydon
"""
import regex as re
import glob
import os
import numpy as np
import pandas as pd

patt_start = re.compile(r'\s(\d{8})\s')
patt_start2 = re.compile(r'\_(\d{4}-\d{2}-\d{2})\_')


input_dir = r"/media/greydon/WD_EXT/emory_ROSA/new"


participant_data = pd.read_excel(os.path.join(input_dir, "EmoryID_master.xlsx"), header=0, sheet_name='patients')


rosa_folders = [x for x in glob.glob(os.path.join(input_dir,"*")) if os.path.isdir(x)]

for idir in rosa_folders:
	date=patt_start.search(os.path.basename(idir))
	if date:
		date_str=date[0].strip()
		datestamp='-'.join([date_str[:4],date_str[4:6],date_str[6:]])
		name_str=os.path.basename(idir)[:date.span()[0]].strip()
		namestamp=[x.title() for x in re.split('\s+', name_str)]
		new_name=os.path.join(os.path.dirname(idir),'_'.join(namestamp+[datestamp,'ROSA']))
		if not os.path.exists(new_name):
			os.rename(idir, new_name)

for idir in rosa_folders:
	date2=patt_start2.search(os.path.basename(idir))
	if date2:
		name_str=os.path.basename(idir)[:date2.span()[0]].strip()
		lastname=name_str.split('_')[0].lower()
		firstname=', '.join(name_str.split('_')[1:]).lower()
		sub_data=participant_data[(participant_data['lastname'].str.lower()==lastname) & (participant_data['firstname'].str.lower()==firstname)]
		if sub_data.shape[0]>0:
			pt_idx=participant_data[(participant_data['lastname'].str.lower()==lastname) & (participant_data['firstname'].str.lower()==firstname)].index[0]
			participant_data.loc[pt_idx,'rosa']=1


participant_data.to_excel(os.path.join(input_dir, "EmoryID_master.xlsx"),sheet_name='patients')
