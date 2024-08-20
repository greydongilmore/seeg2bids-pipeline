from nilearn import plotting, image
import matplotlib.pyplot as plt
import matplotlib
import nibabel as nib
import numpy as np
import base64
import os
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
from scipy.ndimage import affine_transform




def svg2str(display_object, dpi):
	"""Serialize a nilearn display object to string."""
	from io import StringIO

	image_buf = StringIO()
	display_object.frame_axes.figure.savefig(
		image_buf, dpi=dpi, format="svg", facecolor="k", edgecolor="k"
	)
	return image_buf.getvalue()

def extract_svg(display_object, dpi=300):
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
@keyframes flickerAnimation%s { 0%% {opacity: 2;} 100%% { opacity:0; }}
.foreground-svg { animation: 2s ease-in-out 0s alternate none infinite running flickerAnimation%s;}
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
	
	isub="017"
	datap=r'/media/greydon/lhsc_data/SEEG_peds/derivatives/atlasreg/'
	
	input=dotdict({
		'flo':datap+f'sub-P{isub}/sub-P{isub}_space-T1w_desc-rigid_ct.nii.gz',
		#'flo':datap+f'sub-P{isub}/sub-P{isub}_space-MNI152NLin2009cSym_desc-affine_T1w.nii.gz',
		'ref':datap+f'sub-P{isub}/sub-P{isub}_acq-contrast_T1w.nii.gz'
		#'ref':'/home/greydon/Documents/GitHub/seeg2bids-pipeline/resources/tpl-MNI152NLin2009cSym/tpl-MNI152NLin2009cAsym_res-1_T1w.nii.gz'
	})
	
	output=dotdict({
		'html':'/home/greydon/Downloads/' + f'sub-P{isub}_from-ct_to-noncontrast_regqc.html',
		#'html':'/home/greydon/Downloads/' + f'sub-P{isub}_desc-affine_from-subject_to-MNI152NLin2009cSym_regqc.html',
		'png':'/home/greydon/Downloads/' + f'sub-P{isub}_from-ct_to-noncontrast_regqc.png'
		#'png':'/home/greydon/Downloads/' + f'sub-P{isub}_desc-affine_from-subject_to-MNI152NLin2009cSym_regqc.png'
	})
	
	snakemake = Namespace(output=output, input=input)

	title = 'sub-P017'


#%%


ref_img=nib.load(snakemake.input.ref)
#ref_img.header.set_data_dtype('float32')
ref_img.set_qform(ref_img.affine,1)
ref_img_resamp = image.resample_img(ref_img, target_affine=np.eye(3), interpolation='linear')

flo_img=nib.load(snakemake.input.flo)
#flo_img.header.set_data_dtype('float32')
flo_img.set_qform(flo_img.affine,1)
flo_resamp = image.resample_to_img(flo_img, ref_img_resamp, interpolation='linear')


#mean_mm2vox = np.linalg.inv(ref_img_resamp.affine)
#struct_vox2mean_vox = mean_mm2vox @ ref_img_resamp.affine
#mat, vec = nib.affines.to_matvec(struct_vox2mean_vox)
#resampled_mean = affine_transform(flo_resamp.get_fdata(), mat, vec, output_shape=ref_img_resamp.shape)
#resampled_mean = nib.Nifti1Image(resampled_mean, header=flo_resamp.header, affine=ref_img_resamp.affine)
#
#flo_resamp_vox_center = (np.array(resampled_mean.shape) - 1) / 2.
#flo_cut=(resampled_mean.affine.dot(list(flo_resamp_vox_center) + [1])).astype(int).tolist()


plot_args_ref={'dim':-1, 'black_bg':True}
plot_args_flo={'dim':0, 'black_bg':False}
if any(x in os.path.basename(snakemake.output.png) for x in ['from-subject_to-']):
	plot_args_ref={'dim':1}


if any(x in os.path.basename(snakemake.output.png) for x in ['from-ct']):
	plot_args_ref={'dim':-.5}
	plot_args_flo={'dim':.5,'vmin': 300,'vmax':1800}

final_svg=[]
for icut in (-40,-20,0,20,40):
	display = plotting.plot_anat(ref_img_resamp, display_mode='ortho', cut_coords=[0,icut,icut], draw_cross=False, **plot_args_ref)
	fg_svgs = [fromstring(extract_svg(display,300))]
	display.close()
	
	display = plotting.plot_anat(flo_resamp, display_mode='ortho', draw_cross=False,cut_coords=[0,icut,icut], **plot_args_flo)
	bg_svgs = [fromstring(extract_svg(display,300))]
	display.close()
	
	final_svg.append(clean_svg(fg_svgs, bg_svgs))

final_svg="\n".join(sum(final_svg,[]))


# make figure of thalamic contours
display = plotting.plot_anat(ref_img_resamp, display_mode='ortho',draw_cross=False,cut_coords=[0,0,20],**plot_args_ref)
display.add_contours(flo_resamp,alpha=0.7,colors='r',linewidths=0.8)
display.savefig(snakemake.output.png,dpi=350)

# tmpfile_ref = BytesIO()
# display.savefig(tmpfile_ref,dpi=300)
# display.close()
# tmpfile_ref.seek(0)
# data_uri = base64.b64encode(tmpfile_ref.getvalue()).decode('utf-8')
# img_tag = '<center><img src="data:image/png;base64,{0}"/></center>'.format(data_uri)


htmlbase='<!DOCTYPE html> <html lang="en"> <head> <title>Slice viewer</title>  <meta charset="UTF-8" /> </head> <body>'
htmlend='</body> </html>'

htmlfull=htmlbase + final_svg  + htmlend

# Write HTML String to file.html
with open(snakemake.output.html, "w") as file:
	file.write(htmlfull)
	

	