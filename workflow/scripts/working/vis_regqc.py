from nilearn import plotting, image
from nilearn.datasets import load_mni152_template
import matplotlib.pyplot as plt
import matplotlib
import nibabel as nib
from io import BytesIO
import base64
from svgutils.transform import SVGFigure, GroupElement,fromstring
from svgutils.compose import Unit
from tempfile import TemporaryDirectory
from pathlib import Path
from uuid import uuid4
import re
import numpy as np
from matplotlib import gridspec


#matplotlib.use('Qt5Agg')

def svg2str(display_object, dpi):
	"""Serialize a nilearn display object to string."""
	from io import StringIO

	image_buf = StringIO()
	display_object.frame_axes.figure.savefig(
		image_buf, dpi=dpi, format="svg", facecolor="k", edgecolor="k"
	)
	return image_buf.getvalue()

def extract_svg(display_object, dpi=400):
	"""Remove the preamble of the svg files generated with nilearn."""
	image_svg = svg2str(display_object, dpi)
	
	image_svg = re.sub(' height="[0-9]+[a-z]*"', "", image_svg, count=1)
	image_svg = re.sub(' width="[0-9]+[a-z]*"', "", image_svg, count=1)
	image_svg = re.sub(
		" viewBox", ' preseveAspectRation="xMidYMid meet" viewBox', image_svg, count=1
	)
	start_tag = "<svg "
	start_idx = image_svg.find(start_tag)
	end_tag = "</svg>"
	end_idx = image_svg.rfind(end_tag)
	
	# rfind gives the start index of the substr. We want this substr
	# included in our return value so we add its length to the index.
	end_idx += len(end_tag)
	return image_svg[start_idx:end_idx]

def clean_svg(fg_svgs, bg_svgs, ref=0):
	# Find and replace the figure_1 id.
	svgs = bg_svgs+fg_svgs
	roots = [f.getroot() for f in svgs]
	
	sizes = []
	for f in svgs:
		viewbox = [float(v) for v in f.root.get("viewBox").split(" ")]
		width = int(viewbox[2])
		height = int(viewbox[3])
		sizes.append((width, height))
	nsvgs = len([bg_svgs])
	
	sizes = np.array(sizes)

	# Calculate the scale to fit all widths
	width = sizes[ref, 0]
	scales = width / sizes[:, 0]
	heights = sizes[:, 1] * scales

	# Compose the views panel: total size is the width of
	# any element (used the first here) and the sum of heights
	fig = SVGFigure(Unit(f"{width}px"), Unit(f"{heights[:nsvgs].sum()}px"))

	yoffset = 0
	for i, r in enumerate(roots):
		r.moveto(0, yoffset, scale_x=scales[i])
		if i == (nsvgs - 1):
			yoffset = 0
		else:
			yoffset += heights[i]
	
	# Group background and foreground panels in two groups
	if fg_svgs:
		newroots = [
			GroupElement(roots[:nsvgs], {"class": "background-svg"}),
			GroupElement(roots[nsvgs:], {"class": "foreground-svg"}),
		]
	else:
		newroots = roots
	
	fig.append(newroots)
	fig.root.attrib.pop("width", None)
	fig.root.attrib.pop("height", None)
	fig.root.set("preserveAspectRatio", "xMidYMid meet")

	with TemporaryDirectory() as tmpdirname:
		out_file = Path(tmpdirname) / "tmp.svg"
		fig.save(str(out_file))
		# Post processing
		svg = out_file.read_text().splitlines()

	# Remove <?xml... line
	if svg[0].startswith("<?xml"):
		svg = svg[1:]

	# Add styles for the flicker animation
	if fg_svgs:
		svg.insert(
			2,
			"""\
<style type="text/css">
@keyframes flickerAnimation%s { 0%% {opacity: 1;} 100%% { opacity:0; }}
.foreground-svg { animation: 1.5s ease-in-out 0s alternate none infinite paused flickerAnimation%s;}
.foreground-svg:hover { animation-play-state: running;}
</style>"""
			% tuple([uuid4()] * 2),
		)
	
	return svg

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
	
	isub="107"
	datapath=r'/media/greydon/lhsc_data/SEEG_rerun/derivatives/atlasreg'
	input=dotdict({
		'flo':'/media/greydon/lhsc_data/SEEG_rerun/derivatives/atlasreg/sub-P107/sub-P107_space-MNI152NLin2009aAsym_desc-SyN_T1w.nii.gz',
		'ref': '/home/greydon/Documents/GitHub/seeg2bids-pipeline/resources/tpl-MNI152NLin2009aAsym/tpl-MNI152NLin2009aAsym_res-1_T1w.nii.gz'
	})
	
	output=dotdict({
		'html':'/home/greydon/Downloads/' + f'sub-P{isub}_desc-masked_from-ct_to-T1w_regqc.html',
		'png':'/home/greydon/Downloads/' + f'sub-P{isub}_desc-masked_from-ct_to-T1w_regqc.png'
	})
	
	snakemake = Namespace(output=output, input=input)

	title = 'sub-P001'


def add_pad(image, new_height=512, new_width=512):
	height, width,depth = image.shape
	new_dim = max(height, width)
	final_image = np.zeros((new_dim, new_dim, depth))

	pad_left = int((new_dim - width) / 2)
	pad_top = int((new_dim - height) / 2)
	
	# Replace the pixels with the image's pixels
	final_image[pad_top:pad_top + height, pad_left:pad_left + width,:] = image
	
	return final_image

def sample_stack(img_data_raw, show_every=5):
	img_data = np.fliplr(np.rot90(img_data_raw.copy(),1))
	img_data = add_pad(img_data)
	z_slices = np.arange(20,img_data.shape[2]-20, show_every)
	num_rc=int(np.floor(np.sqrt(len(z_slices))))
	#fig,ax = plt.subplots(num_rc,num_rc,figsize=[12,12],gridspec_kw = {'wspace':0, 'hspace':0})
	fig = plt.figure(figsize=(12,12)) 
	cnt = 0
	gs = gridspec.GridSpec(num_rc, num_rc, wspace=0, hspace=0)
	for i in range(num_rc*num_rc):
		
		if i >= len(z_slices):
			#fig.delaxes(ax[int(i/rows),int(i % rows)])
			continue
		else:
			ax= plt.subplot(gs[int(i/num_rc),int(i % num_rc)])
			ind = int(z_slices[i])
			ax.imshow(img_data[:,:,ind],cmap='gray')
			ax.axis('off')
			ax.set_aspect('equal')
	
	plt.tight_layout()
	return plt

#%%

template = load_mni152_template()

ref_img=nib.load(snakemake.input.ref)
flo_img=nib.load(snakemake.input.flo)
brain_img=nib.load(r'/home/greydon/Documents/data/DBS/derivatives/fastsurfer/sub-P237/fastsurfer/mri/brain.mgz')

ref_resample = image.resample_img(ref_img, target_affine=np.eye(3), interpolation='nearest')
flo_resample = image.resample_img(flo_img, target_affine=np.eye(3), interpolation='nearest')
brain_resample = image.resample_img(brain_img, target_affine=np.eye(3), interpolation='nearest')

#flo_resamp = image.resample_to_img(snakemake.input.flo, ref_resamp)

title='sub-{subject}'.format(**snakemake.wildcards)

kwargs = {"linewidths":.8}

display = plotting.plot_anat(ref_resample, display_mode='ortho',cut_coords=[0,0,0])
display.add_edges(flo_img)
display.savefig(snakemake.output.png, dpi=250)
display.close()

#html_view.save_as_html(snakemake.output.html)
html_out=[]
for iz in [-40]:
	display = plotting.plot_anat(ref_resample, display_mode='ortho',draw_cross=False, cut_coords=[0,0,iz])
	fg_svgs = [fromstring(extract_svg(display))]
	display.close()
	
	display = plotting.plot_anat(flo_resample, display_mode='ortho',draw_cross=False, cut_coords=[0,0,iz])
	bg_svgs = [fromstring(extract_svg(display))]
	display.close()
	
	final_svg="\n".join(clean_svg(bg_svgs,fg_svgs,1))
	
	html_out.append([f"""
			<center>
				<h1 style="font-size:42px">{isub}</h1>
				<p>{final_svg}</p>
				<hr style="height:4px;border-width:0;color:black;background-color:black;margin:30px;">
			</center>"""])

html_string=''.join([x[0] for x in html_out])


with open(snakemake.output.html,'w') as fid:
	fid.write(html_string)


ref_view=sample_stack(ref_resample.get_fdata())
flo_view=sample_stack(flo_resample.get_fdata())

with TemporaryDirectory() as tmpdirname:
	out_file = Path(tmpdirname) / "tmp1.svg"
	ref_view.savefig(str(out_file))
	svg_ref = out_file.read_text().splitlines()

with TemporaryDirectory() as tmpdirname:
	out_file = Path(tmpdirname) / "tmp1.svg"
	flo_view.savefig(str(out_file))
	svg_flo = out_file.read_text().splitlines()

svg_ref.insert(
			2,
			"""\
<style type="text/css">
@keyframes flickerAnimation%s { 0%% {opacity: 1;} 100%% { opacity:0; }}
.foreground-svg { animation: 1.5s ease-in-out 0s alternate none infinite paused flickerAnimation%s;}
.foreground-svg:hover { animation-play-state: running;}
</style>"""
			% tuple([uuid4()] * 2),
		)

svg_combine=svg_ref+svg_flo


final_svg="\n".join(svg_combine)
tmpfile_ref = BytesIO()
ref_view.savefig(tmpfile_ref,dpi=300)
tmpfile_ref.seek(0)
encoded_ref = base64.b64encode(tmpfile_ref.getvalue())

tmpfile_flo = BytesIO()
flo_view.savefig(tmpfile_flo,dpi=300)
tmpfile_flo.seek(0)
encoded_flo = base64.b64encode(tmpfile_flo.getvalue())

html_list=[f"""
		<center>
			<h1 style="font-size:42px">{isub}</h1>
			<p><img src="data:image/png;base64, {encoded.decode("utf-8")}" width=1200 height=1200></p>
			<hr style="height:4px;border-width:0;color:black;background-color:black;margin:30px;">
		</center>"""]
		
html_view.close()

html_out_01="""
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" viewBox="0 0 475.0 187.0" preserveAspectRatio="xMidYMid meet">
   
    <style type="text/css">
     @keyframes flickerAnimation67aaf783-cc13-4a0a-9ee8-5b9dbd3abb1a { 0% {opacity: 1;} 100% { opacity:0; }}
     .foreground-svg { animation: 1.5s ease-in-out 0s alternate none infinite paused flickerAnimation67aaf783-cc13-4a0a-9ee8-5b9dbd3abb1a;}
     .foreground-svg:hover { animation-play-state: running;}
   </style>
"""
html_out_02 = """
  <defs>
    <style type="text/css">*{stroke-linecap:butt;stroke-linejoin:round;}</style>
  </defs>
""" +\
f"""
  <g id="figure_1">
    <g class="background-svg" height="187.2" width="475.2">
      <image height="187.2" id="imagea1" width="475.2" x="0" xlink:href="data:image/png;base64,{encoded_ref.decode("utf-8")}></image>
    </g>
  </g>
""" +\
"""
  <defs>
    <style type="text/css">*{stroke-linecap:butt;stroke-linejoin:round;}</style>
  </defs>
""" +\
f"""
  <g id="figure_1">
    <g class="foreground-svg" height="187.2" width="475.2">
      <image height="187.2" id="image2" width="475.2" x="0" xlink:href="data:image/png;base64,{encoded_flo.decode("utf-8")}></image>
    </g>
  </g>
</svg>
"""


html_string='\n'.join([html_out_01, html_out_02])

with open(snakemake.output.html,'w') as fid:
	fid.write(html_string)


with open(snakemake.output.html,'w') as fid:
	fid.write(final_svg)




from mayavi import mlab
from skimage import measure

mlab.figure(bgcolor=(0, 0, 0), size=(400, 400))

mlab.contour3d(brain_resample.get_fdata())

p = brain_resample.get_fdata().transpose(2,1,0)
verts, faces, norm, val = measure.marching_cubes(p, step_size=1, allow_degenerate=True)
xx, yy, zz = verts
mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], faces)

src = mlab.pipeline.brain_resample(ref_img.get_fdata())
# Our data is not equally spaced in all directions:
src.spacing = [1, 1, 1.5]
src.update_image_data = True


# Extract some inner structures: the ventricles and the inter-hemisphere
# fibers. We define a volume of interest (VOI) that restricts the
# iso-surfaces to the inner of the brain. We do this with the ExtractGrid
# filter.
blur = mlab.pipeline.user_defined(src, filter='ImageGaussianSmooth')
voi = mlab.pipeline.extract_grid(blur)
voi.trait_set(x_min=125, x_max=193, y_min=92, y_max=125, z_min=34


mlab.show()