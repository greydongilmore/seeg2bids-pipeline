
import pandas as pd
import numpy as np
import matplotlib
import re,os
import matplotlib.pyplot as plt
from nilearn import plotting
import cv2
from PIL import Image

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
	
	isub="P162"
	data_dir=r'/home/greydon/Documents/data/SEEG/derivatives'
	
	input=dotdict({'fcsv':f'{data_dir}/seeg_coordinates/' + f'sub-{isub}/sub-{isub}_space-native_SEEGA.fcsv',
				'xfm_ras':f'{data_dir}/atlasreg/' + f'sub-{isub}/sub-{isub}_desc-affine_from-subject_to-MNI152NLin2009cSym_type-ras_xfm.txt'
				})
	
	output=dotdict({'html':f'{data_dir}/atlasreg/' + f'sub-{isub}/qc/sub-{isub}_space-MNI152NLin2009cSym_desc-affine_electrodes.html',
				'png':f'{data_dir}/atlasreg/' + f'sub-{isub}/qc/sub-{isub}_space-MNI152NLin2009cSym_desc-affine_electrodevis.png'
				})
	
	snakemake = Namespace(output=output, input=input)

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
	vals=vals[indexes.argsort()]
	count=count[indexes.argsort()]
	return vals,count

#read fcsv electrodes file
df = pd.read_table(snakemake.input.fcsv,sep=',',header=2)

groups,n_members=determine_groups(df['label'].tolist(),numbered_labels=True)
df['group']=np.repeat(groups,n_members)

cmap = plt.get_cmap('rainbow')
color_maps=cmap(np.linspace(0, 1, len(groups))).tolist()
res = dict(zip(groups, color_maps))

colors=[]
for igroup in df['group']:
	colors.append(res[igroup])

colors=np.vstack(colors)

labels=[str(x) for x in  range(colors.shape[0])]

#load transform from subj to template
sub2template= np.loadtxt(snakemake.input.xfm_ras)

#plot electrodes transformed (affine) to MNI space, with MNI glass brain
coords = df[['x','y','z']].to_numpy()

#to plot in mni space, need to transform coords
tcoords = np.zeros(coords.shape)
for i in range(len(coords)):

    vec = np.hstack([coords[i,:],1])
    tvec = sub2template @ vec.T
    tcoords[i,:] = tvec[:3]

html_view = plotting.view_markers(tcoords, marker_size=4.0, marker_color=colors, marker_labels=df['label'].tolist())
#html_view.open_in_browser()
html_view.save_as_html(snakemake.output.html)

#plot subject native space electrodes with glass brain
adjacency_matrix = np.zeros([len(coords),len(coords)])

node_label=np.repeat(groups, n_members, axis=0)

group = np.array([1,3,2,1,3])
cdict = {1: 'red', 2: 'blue', 3: 'green'}

_, idx = np.unique(colors, return_index=True, axis=0)

label_dict=dict(zip(groups,colors[np.sort(idx)].tolist()))

temp1=''.join([os.path.splitext(snakemake.output.png)[0],'_1',os.path.splitext(snakemake.output.png)[-1]])
temp2=''.join([os.path.splitext(snakemake.output.png)[0],'_2',os.path.splitext(snakemake.output.png)[-1]])

display = plotting.plot_connectome(adjacency_matrix, tcoords, node_color=colors, node_size=3)
display.savefig(temp1,dpi=300)
display.close()

legend_elements = []
for k,v in label_dict.items():
	legend_elements.append(matplotlib.lines.Line2D([0], [0], marker='o', color=v, label=k,markerfacecolor=v, markersize=6,linestyle='none'))

# Create the figure
fig, ax = plt.subplots()
ax.axis("off")
ax.legend(handles=legend_elements, loc='center')
fig.savefig(temp2,dpi=300)
plt.close()

img_orig = cv2.imread(temp2)
img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
thresh = img.copy()
thresh[img ==255] = 0

contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
big_contour = max(contours, key=cv2.contourArea)
mask = np.zeros_like(thresh, dtype=np.uint8)
cv2.drawContours(mask, [big_contour], 0, 255, -1)
x,y,w,h = cv2.boundingRect(big_contour)

img_crop=img_orig.copy()
img_crop[mask==0]=0
img_crop = img_crop[y:y+h, x:x+w]
image_1 = cv2.imread(temp1)

new_width=(image_1.shape[0]/img_crop.shape[0])*img_crop.shape[1]
image_1 = cv2.imread(temp1)
img_crop=cv2.resize(img_crop, (int(new_width), image_1.shape[0]))
cv2.imwrite(temp2, img_crop)


images = [Image.open(x) for x in [temp1, temp2]]
widths, heights = zip(*(i.size for i in images))
total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width+20, max_height),"WHITE")
x_offset=0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]+20

new_im.save(snakemake.output.png)

os.remove(temp1)
os.remove(temp2)
