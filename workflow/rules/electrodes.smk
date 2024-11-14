
rule electrode_coords:
    input:
        seega_scene = get_electrodes_coords(subject_id,coords_type='SEEGA'),
    params:
        sub=subject_id
    output:
        seega_fcsv = f'{sep}'.join([config['out_dir'], config['seeg_contacts']['space_coords'].format(subject=subject_id, coords_space='native', coords_type='SEEGA')])
    group: 'preproc'
    script: '../scripts/working/elec_labels_coords.py'

final_outputs.extend(expand(f'{sep}'.join([config['out_dir'], config['seeg_contacts']['space_coords'].format(subject=subject_id, coords_space='native', coords_type='SEEGA')]),
        subject=subjects))

if config['segmentation']['run']:
    rule warp_contact_coords:
        input: 
            fcsv = get_electrodes_coords(subject_id,coords_space='native', coords_type='SEEGA'),
            xfm_composite = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='InverseComposite.h5',from_='subject',to=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space')),
        output:
            fcsv_fname_warped = f'{sep}'.join([config['out_dir'], config['seeg_contacts']['space_coords'].format(subject=subject_id, coords_space=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'), coords_type='SEEGA')])
        group: 'preproc'
        script: '../scripts/working/apply_warp_to_points.py'

    rule label_electrodes_atlas:
        input: 
            fcsv = get_electrodes_coords(subject_id,coords_space='native', coords_type='SEEGA'),
            dseg_tsv = get_age_appropriate_template_name(expand(subject_id,subject=subjects),'atlas_dseg_tsv'),
            dseg_nii = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='dseg.nii.gz', atlas='{atlas}',from_=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='nonlin',label='dilated'),
            tissue_seg = expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='probseg.nii.gz',label='{tissue}',desc='atropos3seg'),
                                tissue=config['tissue_labels'],allow_missing=True),
        output:
            tsv = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='electrodes.tsv',atlas='{atlas}', from_=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='nonlin'),
    #        tsv = report(bids(root='results',subject=subject_id,suffix='electrodes.tsv',desc='{atlas}',from_='{template}'),
    #                caption='../reports/electrodes_vis.rst',
    #                category='Electrodes Labelled',
    #                subcategory='Atlas: {atlas}, Template: {template}')           
        group: 'preproc'
        script: '../scripts/label_electrodes_atlas.py'

rule contact_landmarks:
    input: 
        fcsv = get_electrodes_coords(subject_id,coords_space='native', coords_type='SEEGA'),
    output:
        txt = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='landmarks.txt',space=config['post_image']['suffix']),    
    group: 'preproc'
    run:
        df = pd.read_table(input.fcsv,sep=',',header=2)
        coords = df[['x','y','z']].to_numpy()
        with open (output.txt, 'w') as fid:
            for i in range(len(coords)):
                fid.write(' '.join(str(i) for i in np.r_[np.round(coords[i,:],3),int(1)])+ "\n")

rule mask_contacts:
    input: 
        ct = bids(root=join(config['out_dir'],'derivatives', 'atlasreg'),subject=subject_id,suffix=config['post_image']['suffix']+config['post_image']['ext'],space='T1w',desc='rigid',ses='post',include_session_dir=False),
        txt = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,suffix='landmarks.txt',space=config['post_image']['suffix']),
    params:
        c3d=config['ext_libs']['c3d'],
    output:
        mask = bids(root=join(config['out_dir'],'derivatives', 'atlasreg'),subject=subject_id,suffix='contacts.nii.gz',space=config['post_image']['suffix'],desc='mask'),
    group: 'preproc'
    shell:
        '{params.c3d} {input.ct} -scale 0 -landmarks-to-spheres {input.txt} 1 -o {output.mask}'

rule vis_contacts:
    input:
        ct = bids(root=join(config['out_dir'],'derivatives', 'atlasreg'),subject=subject_id,suffix=config['post_image']['suffix']+config['post_image']['ext'],space='T1w',desc='rigid',ses='post',include_session_dir=False),
        mask = bids(root=join(config['out_dir'],'derivatives', 'atlasreg'),subject=subject_id,suffix='contacts.nii.gz',space=config['post_image']['suffix'],desc='mask'),
    output:
        html = report(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='contacts.html',desc='mask',space=config['post_image']['suffix'],include_subject_dir=False),
                caption='../reports/contacts_vis.rst',
                category='Contacts in CT space',
                subcategory='landmarks mask'),
    group: 'preproc'
    script: '../scripts/vis_contacts.py'

rule vis_electrodes:
    input: 
        fcsv = get_electrodes_coords(subject_id,coords_space='native',coords_type='SEEGA'),
        t1w = bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject=subject_id,desc='n4', suffix='T1w.nii.gz'),
        xfm_ras = bids(root=join(config['out_dir'],'derivatives', 'atlasreg'),subject=subject_id,suffix='xfm.txt',from_='subject',to=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),desc='affine',type_='ras'),
    params:
        contacts= bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='contacts.html',desc='mask',space=config['post_image']['suffix'],include_subject_dir=False)
    output:
        html = report(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='electrodes.html',desc='affine',space=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space')),
                caption='../reports/electrodes_vis.rst',
                category='Electrodes in template space',
                subcategory=f"reg to {get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space')}"),
        png = report(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='electrodevis.png',desc='affine',space=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),include_subject_dir=False),
                caption='../reports/electrodes_vis.rst',
                category='Electrodes in template space',
                subcategory=f"reg to {get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space')}"),
    group: 'preproc'
    script: '../scripts/vis_electrodes.py'

final_outputs.extend(expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='electrodevis.png',desc='affine',space=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),include_subject_dir=False),
                    subject=subjects, desc=['rigid']))

final_outputs.extend(expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='electrodes.html',desc='affine',space=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),include_subject_dir=False),
                    subject=subjects, desc=['rigid']))

final_outputs.extend(expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='contacts.html',desc='mask',space=config['post_image']['suffix'],include_subject_dir=False),
        subject=subjects))

if config['segmentation']['run']:
    final_outputs.extend(
        expand(
            rules.warp_contact_coords.output.fcsv_fname_warped,
            subject=subjects
        )
    )

    final_outputs.extend(
        expand(
            bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),
                subject=subject_id,
                suffix='electrodes.tsv',
                atlas='{atlas}',
                from_=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space'),
                desc='nonlin'
            ),
            subject=subjects,
            atlas=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'atlas'),
            template=get_age_appropriate_template_name(expand(subject_id,subject=subjects),'space')
        )
    )


