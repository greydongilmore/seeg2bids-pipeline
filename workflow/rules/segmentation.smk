

def get_k_tissue_classes(wildcards):
    k_classes=config['default_k_tissue_classes']
    return k_classes


#this performs Atropos with k-means as initialization
rule tissue_seg_kmeans_init:
    input:
        t1 = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'), subject=subject_id,desc='n4', suffix='T1w.nii.gz'),
        mask = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='mask.nii.gz',from_=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='affine',label='brain'),
    params:
        k = config['default_k_tissue_classes'],
        m = config['atropos_smoothing_factor'],
        c = config['convergence'],
        posterior_fmt = 'posteriors_%d.nii.gz',
        posterior_glob = 'posteriors_*.nii.gz',
        fslmerge=config['ext_libs']['fslmerge'],
    output:
        seg = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='dseg.nii.gz',desc='atroposKseg'),
        posteriors = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='probseg.nii.gz',desc='atroposKseg'),
    shadow: 'minimal'
    #container: config['singularity']['neuroglia']
    group: 'preproc'
    threads: 6
    shell:
        #'Atropos -d 3 -a {input.t1} -i kmeans[{params.k}] -m {params.m} -c {params.c} -x {input.mask} -o [{output.seg},{params.posterior_fmt}] && '
        'Atropos -d 3 -a {input.t1} -i KMeans[{params.k}] -x {input.mask} -o [{output.seg},{params.posterior_fmt}] && '
        '{params.fslmerge} -t {output.posteriors} {params.posterior_glob} ' #merge posteriors into a 4d file (intermediate files will be removed b/c shadow)

rule map_channels_to_tissue:
    input:
        tissue_priors = expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='probseg.nii.gz',label='{tissue}',from_=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='affine'),
                            tissue=config['tissue_labels'],allow_missing=True),
        seg_channels_4d = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='probseg.nii.gz',desc='atroposKseg'),
    output:
        mapping_json = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='mapping.json',desc='atropos3seg'),
        tissue_segs = expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='probseg.nii.gz',label='{tissue}',desc='atropos3seg'),
                            tissue=config['tissue_labels'],allow_missing=True),
    group: 'preproc'
    script: '../scripts/map_channels_to_tissue.py'

rule tissue_seg_to_4d:
    input:
        tissue_segs = expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='probseg.nii.gz',label='{tissue}',desc='atropos3seg'),
                            tissue=config['tissue_labels'],allow_missing=True),
    params:
        fslmerge=config['ext_libs']['fslmerge'],
    output:
        tissue_seg = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='probseg.nii.gz',desc='atropos3seg')
    group: 'preproc'
    #container: config['singularity']['neuroglia']
    shell:
        '{params.fslmerge} -t {output} {input}'

rule brainmask_from_tissue:
    input:
        tissue_seg = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='probseg.nii.gz',desc='atropos3seg')
    params:
        threshold = 0.5,
        fslmaths=config['ext_libs']['fslmaths'],
    output:
        mask = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='mask.nii.gz',from_='atropos3seg',desc='brain')
    #container: config['singularity']['neuroglia']
    group: 'preproc'
    shell:
        #max over tissue probs, threshold, binarize, fill holes
       '{params.fslmaths} {input} -Tmax -thr {params.threshold} -bin -fillh {output}'     

rule tissue_warp_to_nrrd:
    input:
        segs = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='dseg.nii.gz',atlas='{atlas}',from_=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='nonlin'),
    params:
        atlas_labels= get_age_appropriate_template_name(expand(subject_id,subject=subjects),'atlas_dseg_tsv'),
        atlas_colors= config['generic_colors'],
        orien= get_age_appropriate_template_name(expand(subject_id,subject=subjects),'orien'),
    output:
        seg_nrrd = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='dseg.seg.nrrd',atlas='{atlas}',from_=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='nonlin'),
    group: 'preproc'
    #container: config['singularity']['neuroglia']
    script: '../scripts/working/niiTonrrd.py' 

rule tissue_4d_to_nrrd:
    input:
        segs = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='dseg.nii.gz',atlas='{atlas}',from_=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='nonlin',label='dilated')
    params:
        atlas_labels= get_age_appropriate_template_name(expand(subject_id,subject=subjects),'atlas_dseg_tsv'),
        atlas_colors= config['generic_colors'],
        orien= get_age_appropriate_template_name(expand(subject_id,subject=subjects),'orien'),
    output:
        seg_nrrd = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='dseg.seg.nrrd',atlas='{atlas}',from_=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='nonlin',label='dilated')
    group: 'preproc'
    #container: config['singularity']['neuroglia']
    script: '../scripts/working/niiTonrrd.py'

#rule gradient_magnitude:
#    input:
#        t1 = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'), subject=subject_id,desc='n4', suffix='T1w.nii.gz'),
#        gm = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='probseg.nii.gz',label='GM',desc='atropos3seg'),
#        wm = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='probseg.nii.gz',label='WM',desc='atropos3seg'),
#        mask = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='mask.nii.gz',from_=config[get_age_appropriate_template_name(subject_id)].format(template=config['template']),reg='affine',desc='brain'),
#        seg = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='dseg.nii.gz',desc='atroposKseg'),
#    output:
#        t1_grad = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id, desc='magnitude', suffix='T1w.nii.gz'),
#        t1_grad_color = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id, desc='magnitude', label='hot', suffix='T1w.nii.gz'),
#        t1_intensity = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id, desc='relintensity', suffix='T1w.nii.gz'),
#        t1_intensity_color = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id, desc='relintensity', label='hot', suffix='T1w.nii.gz'),
#        png_mask = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='maskqc.png',desc='brain',include_subject_dir=False),
#        png_seg = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='segqc.png',desc='segmentation',include_subject_dir=False),
#        png_ri = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='qc.png',desc='relintensity',include_subject_dir=False),
#        png_grad = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='qc.png',desc='gradient',include_subject_dir=False),
#    script:
#        '../scripts/grad_mag.py'

#final_outputs.extend(expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='maskqc.png',desc='brain', include_subject_dir=False),
#                            subject=subjects))
#final_outputs.extend(expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='segqc.png',desc='segmentation', include_subject_dir=False),
#                            subject=subjects))
#final_outputs.extend(expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='qc.png',desc='relintensity', include_subject_dir=False),
#                            subject=subjects))
#final_outputs.extend(expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='qc.png',desc='gradient', include_subject_dir=False),
#                            subject=subjects))
#final_outputs.extend(expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id, desc='magnitude', suffix='T1w.nii.gz'), 
#                        subject=subjects))
#final_outputs.extend(expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id, desc='intensity', suffix='T1w.nii.gz'), 
#                        subject=subjects))

final_outputs.extend(expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='dseg.seg.nrrd',atlas='{atlas}',from_=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='nonlin'),subject=subjects, atlas=get_age_appropriate_template_name(subject_id,'atlas')))
final_outputs.extend(expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='dseg.seg.nrrd',atlas='{atlas}',from_=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='nonlin',label='dilated'),subject=subjects, atlas=get_age_appropriate_template_name(subject_id,'atlas')))

#TODO: make lesion mask from the holes in the brainmask (instead of just filling them..) -- could be a nice way to exclude contrast enhanced vessels
