
def get_reference_t1(wildcards):
    if config['contrast_t1']['present']:
        ref_file=expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'), subject='{subject}', acq='contrast', suffix='T1w.nii.gz'),subject=wildcards.subject)
    elif not config['contrast_t1']['present'] and config['noncontrast_t1']['present']:
        ref_file=expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'), subject='{subject}', acq='noncontrast', suffix='T1w.nii.gz'),subject=wildcards.subject)
    return ref_file[0]

if config['segmentation']['run']:
    rule qc_reg_t1:
        input:
            ref = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+"/tpl-"+get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),suffix='T1w.nii.gz', include_subject_dir=False),
            flo = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='T1w.nii.gz', space=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='{desc}'),
        output:
            png = report(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='regqc.png',from_='subject', to=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='{desc}',include_subject_dir=False),
                    caption='../reports/regqc.rst',
                    category='Registration QC',
                    subcategory=f"{get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space')}"),
            html = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='regqc.html',from_='subject', to=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'), desc='{desc}', include_subject_dir=False),
    #        html = report(bids(root='qc',subject=subject_id,suffix='regqc.html',from_='subject', to=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'), desc='{desc}'),
    #                caption='../reports/regqc.rst',
    #                category='Registration QC',
    #                subcategory='{desc} {template}'),
        group: 'preproc'
        script: '../scripts/vis_regqc.py'

    if config['template_reg']['nlin_reg']['algo']=='greedy':
        final_outputs.extend(expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='regqc.png',from_='subject', to=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='{desc}',include_subject_dir=False),
                                subject=subjects, desc=['affine','nonlin']))
    else:
        final_outputs.extend(expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='regqc.png',from_='subject', to=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='{desc}',include_subject_dir=False),
                            subject=subjects, desc=['affine','nonlin']))

if config['contrast_t1']['present'] and config['noncontrast_t1']['present']:
    rule qc_reg_noncontrast:
        input:
            ref = get_reference_t1,
            flo = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,acq='noncontrast',suffix='T1w.nii.gz',space='T1w',desc='rigid'),
        output:
            png = report(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='regqc.png',from_='noncontrast', to='contrast',include_subject_dir=False),
                    caption='../reports/regqc.rst',
                    category='Registration QC',
                    subcategory='{desc} T1w'),
            html = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='regqc.html',from_='noncontrast', to='contrast', include_subject_dir=False),
    #        html = report(bids(root='qc',subject=subject_id,suffix='regqc.html',from_='subject', to=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'), desc='{desc}'),
    #                caption='../reports/regqc.rst',
    #                category='Registration QC',
    #                subcategory='{desc} {template}'),
        group: 'preproc'
        script: '../scripts/vis_regqc.py'

    final_outputs.extend(expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='regqc.png',from_='noncontrast', to='contrast',include_subject_dir=False), 
                        subject=subjects))

if config['post_image']['present']:
    rule qc_reg_post:
        input:
            ref = get_reference_t1,
            flo = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix=config['post_image']['suffix']+config['post_image']['ext'],space='T1w',desc='rigid',ses='post',include_session_dir=False),
        output:
            png = report(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='regqc.png',from_=config['post_image']['suffix'], to='T1w',desc='rigid',ses='post',include_subject_dir=False,include_session_dir=False),
                    caption='../reports/regqc.rst',
                    category='Registration QC',
                    subcategory='{desc} T1w'),
            html = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='regqc.html',from_=config['post_image']['suffix'], to='T1w', desc='rigid',ses='post',include_session_dir=False,include_subject_dir=False),
    #        html = report(bids(root='qc',subject=subject_id,suffix='regqc.html',from_='subject', to=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'), desc='{desc}'),
    #                caption='../reports/regqc.rst',
    #                category='Registration QC',
    #                subcategory='{desc} {template}'),
        group: 'preproc'
        script: '../scripts/vis_regqc.py'

    final_outputs.extend(expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='regqc.png',from_=config['post_image']['suffix'], to='T1w',desc='rigid',ses='post',include_subject_dir=False,include_session_dir=False), 
                        subject=subjects))

if config['pet']['present']:
    rule qc_reg_pet:
        input:
            ref = get_reference_t1,
            flo = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='pet.nii.gz',space='T1w', desc='rigid'),
        output:
            png = report(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='regqc.png',from_='pet', to='T1w',desc='rigid',include_subject_dir=False),
                    caption='../reports/regqc.rst',
                    category='Registration QC',
                    subcategory='{desc} T1w'),
            html = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='regqc.html',from_='pet', to='T1w', desc='rigid',include_subject_dir=False),
    #        html = report(bids(root='qc',subject=subject_id,suffix='regqc.html',from_='subject', to=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'), desc='{desc}'),
    #                caption='../reports/regqc.rst',
    #                category='Registration QC',
    #                subcategory='{desc} {template}'),
        group: 'preproc'
        script: '../scripts/vis_regqc.py'

    final_outputs.extend(expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='regqc.png',from_='pet', to='T1w',desc='rigid',include_subject_dir=False), 
                        subject=subjects))

if config['other_vol']['present']:
    rule qc_reg_other:
        input:
            ref = get_reference_t1,
            flo = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,session=config['other_vol']['session'], suffix=config['other_vol']['suffix']+config['other_vol']['ext'],acq=config['other_vol']['acq'],space='T1w',desc='rigid',include_session_dir=False),
        output:
            png = report(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='regqc.png',acq=config['other_vol']['acq'],from_=config['other_vol']['suffix'], to='T1w',desc='rigid',include_subject_dir=False),
                    caption='../reports/regqc.rst',
                    category='Registration QC',
                    subcategory='{desc} T1w'),
            html = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='regqc.html',acq=config['other_vol']['acq'],from_=config['other_vol']['suffix'], to='T1w', desc='rigid',include_subject_dir=False),
    #        html = report(bids(root='qc',subject=subject_id,suffix='regqc.html',from_='subject', to=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'), desc='{desc}'),
    #                caption='../reports/regqc.rst',
    #                category='Registration QC',
    #                subcategory='{desc} {template}'),
        group: 'preproc'
        script: '../scripts/vis_regqc.py'

    final_outputs.extend(expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='regqc.png',acq=config['other_vol']['acq'],from_=config['other_vol']['suffix'], to='T1w',desc='rigid',include_subject_dir=False), 
                        subject=subjects))
if config['segmentation']['run']:
    rule qc_probseg:
        input:
            img = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'), subject=subject_id, desc='masked', from_='atropos3seg', suffix='T1w.nii.gz'),
            seg4d = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='probseg.nii.gz',desc='atropos3seg'),
            mapping = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='mapping.json',desc='atropos3seg'),
        params:
            ct_png=bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='regqc.png',from_=config['post_image']['suffix'], to='T1w',desc='masked',include_subject_dir=False),
            tissue_classes=config['default_k_tissue_classes'],
        output:
            png = report(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='probseg.png',desc='brainmask',label='atropos3seg',include_subject_dir=False),
                    caption='../reports/segqc.rst',
                    category='Segmentation QC',
                    subcategory='3-class Tissue Segmentation'),
        group: 'preproc'
        script: '../scripts/vis_qc_probseg.py'

    final_outputs.extend(expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='probseg.png', desc='brainmask',label='atropos3seg',include_subject_dir=False),
                            subject=subjects ))

    rule qc_dseg:
        input:
            img = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'), subject=subject_id, desc='masked', from_='atropos3seg', suffix='T1w.nii.gz'),
            seg = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'), subject=subject_id,suffix='dseg.nii.gz',atlas='{atlas}',from_=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='nonlin'),
        output:
            png = report(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='dseg.png',atlas='{atlas}', from_=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='brainmask', include_subject_dir=False),
                    caption='../reports/segqc.rst',
                    category='Segmentation QC',
                    ),
            html = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='dseg.html',atlas='{atlas}', from_=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='brainmask',include_subject_dir=False),
    #        html = report(bids(root='qc',subject=subject_id,suffix='dseg.html',atlas='{atlas}', from_=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space')),
    #                caption='../reports/segqc.rst',
    #                category='Segmentation QC',
    #                subcategory='{atlas} Atlas from {template}'),
        group: 'preproc'
        script: '../scripts/vis_qc_dseg.py'

    final_outputs.extend(expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='dseg.png',atlas='{atlas}', from_=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='brainmask',include_subject_dir=False),
                            subject=subjects, atlas=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'atlas')))

    rule qc_dseg_dilated:
        input:
            img = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'), subject=subject_id, desc='masked', from_='atropos3seg', suffix='T1w.nii.gz'),
            seg = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='dseg.nii.gz',atlas='{atlas}',from_=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='nonlin',label='dilated'),
        output:
            png = report(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='dseg.png',atlas='{atlas}', from_=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='brainmask',label='dilated',include_subject_dir=False),
                    caption='../reports/segqc.rst',
                    category='Segmentation QC'),
            html = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='dseg.html',atlas='{atlas}', from_=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='brainmask',label='dilated',include_subject_dir=False),
    #        html = report(bids(root='qc',subject=subject_id,suffix='dseg.html',atlas='{atlas}', from_=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space')),
    #                caption='../reports/segqc.rst',
    #                category='Segmentation QC',
    #                subcategory='{atlas} Atlas from {template}'),
        group: 'preproc'
        script: '../scripts/vis_qc_dseg.py'

    final_outputs.extend(expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='dseg.png',atlas='{atlas}', from_=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='brainmask',label='dilated',include_subject_dir=False),
                            subject=subjects, atlas=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'atlas')))

    rule qc_tissue_class:
        input:
            img = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'), subject=subject_id, desc='masked', from_='atropos3seg', suffix='T1w.nii.gz'),
            wm = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='probseg.nii.gz',label='WM',desc='atropos3seg'),
            gm = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='probseg.nii.gz',label='GM',desc='atropos3seg'),
            csf = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='probseg.nii.gz',label='CSF',desc='atropos3seg'),
        output:
            png = report(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id, suffix='probseg.png', desc='brainmask', label='classes', include_subject_dir=False),
                    caption='../reports/segqc.rst',
                    category='Segmentation QC',
                    subcategory='Tissue classification'),
        group: 'preproc'
        script: '../scripts/vis_qc_tissue_seg.py'

    final_outputs.extend(expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='probseg.png', desc='brainmask', label='classes',include_subject_dir=False),
                            subject=subjects))

