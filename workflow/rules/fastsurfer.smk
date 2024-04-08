

def get_transform_filename(wildcards):
    file=[]
    if config['contrast_t1']['present'] and config['noncontrast_t1']['present']:
        file=expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),subject='{subject}',suffix='xfm.txt',from_='noncontrast',to='contrast',desc='rigid',type_='ras'),subject=wildcards.subject),
    if len(file) >0:
        file=file[0][0]
    print(file)
    return file

if config['fastsurfer_config']['version'] =='dev':
    rule fastsurfer_seg:
        input: 
            t1 = get_noncontrast_filename_fs,
        params:
            fastsurfer_run = config['fastsurfer_config']['home'],
            sid = config['fastsurfer_config']['sid'],
            batch = config['fastsurfer_config']['batch'],
            threads = config['fastsurfer_config']['threads'],
            vox_size = config['fastsurfer_config']['vox_size'],
            py = config['fastsurfer_config']['py'],
            fastsurfer_out = directory(join(config['out_dir'], 'derivatives', 'fastsurfer')),
            subjid=subject_id,
        output:
            touch_fastsurfer=touch(join(config['out_dir'], 'logs', 'sub-' + subject_id + "_fastsurfer.done")),
            t1_fname = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'mri','orig.mgz'),
            segs = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'mri','aparc+aseg.mgz'),
        group: 'preproc'
        threads: 8
        shell:
            "export FASTSURFER_HOME={params.fastsurfer_run} &&PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096 {params.fastsurfer_run}/run_fastsurfer.sh \
            --t1 {input.t1} --sd {params.fastsurfer_out} --threads {params.threads} --vox_size {params.vox_size} --sid sub-{params.subjid} --py {params.py} --viewagg_device cpu --fsaparc --parallel --allow_root"
elif config['fastsurfer_config']['version'] =='stable':
    rule fastsurfer_seg:
        input: 
            t1 = get_noncontrast_filename_fs(config['fastsurfer_config']['ses']),
        params:
            fastsurfer_run = config['fastsurfer_config']['home'],
            sid = config['fastsurfer_config']['sid'],
            batch = config['fastsurfer_config']['batch'],
            threads = config['fastsurfer_config']['threads'],
            vox_size = config['fastsurfer_config']['vox_size'],
            py = config['fastsurfer_config']['py'],
            fastsurfer_out = directory(join(config['out_dir'], 'derivatives', 'fastsurfer')),
            subjid=subject_id,
        output:
            touch_fastsurfer=touch(join(config['out_dir'], 'logs', 'sub-' + subject_id + "_fastsurfer.done")),
            t1_fname = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'mri','orig.mgz'),
            segs = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'mri','aparc+aseg.mgz'),
        threads: 8
        group: 'preproc'
        shell:
            "export FASTSURFER_HOME={params.fastsurfer_run} &&PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096 {params.fastsurfer_run}/run_fastsurfer.sh \
            --t1 {input.t1} --sd {params.fastsurfer_out} --sid sub-{params.subjid} --py {params.py} --vox_size {params.vox_size} --viewagg_device cpu --fsaparc --no_cereb --parallel --ignore_fs_version --allow_root"
elif config['fastsurfer_config']['version'] =='master':
    rule fastsurfer_seg:
        input: 
            t1 = get_noncontrast_filename_fs(subject_id,config['fastsurfer_config']['ses']),
        params:
            fastsurfer_run = config['fastsurfer_config']['home'],
            sid = config['fastsurfer_config']['sid'],
            batch = config['fastsurfer_config']['batch'],
            threads = config['fastsurfer_config']['threads'],
            py = config['fastsurfer_config']['py'],
            fastsurfer_out = directory(join(config['out_dir'], 'derivatives', 'fastsurfer')),
            subjid=subject_id,
        output:
            touch_fastsurfer=touch(join(config['out_dir'], 'logs', 'sub-' + subject_id + "_fastsurfer.done")),
            t1_fname = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'mri','orig.mgz'),
            segs = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'mri','aparc+aseg.mgz'),
        threads: 8
        group: 'preproc'
        shell:
            "export FASTSURFER_HOME={params.fastsurfer_run} &&PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096 {params.fastsurfer_run}/run_fastsurfer.sh \
            --t1 {input.t1} --sd {params.fastsurfer_out} --sid sub-{params.subjid} --py {params.py} --run_viewagg_on cpu --fsaparc --parallel"

#final_outputs.extend(expand(rules.fastsurfer_seg.output.touch_fastsurfer, subject=subjects))

rule fastsurfer_symlinks:
    input: 
        t1_fname = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'mri','orig.mgz'),
        touch_fastsurfer = rules.fastsurfer_seg.output.touch_fastsurfer,
    params:
        talairach_xfm = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'mri','transforms','talairach.xfm.lta'),
        lh_pial_t1 = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'surf','lh.pial.T1'),
        rh_pial_t1 = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'surf','rh.pial.T1'),
        lh_white_preaparc_h = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'surf','lh.white.preaparc.H'),
        rh_white_preaparc_h = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'surf','rh.white.preaparc.H'),
        lh_white_preaparc_k = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'surf','lh.white.preaparc.K'),
        rh_white_preaparc_k = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'surf','lh.white.preaparc.K'),
        talairach_lta = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'mri','transforms','talairach.lta'),
        talairach_skull_lta = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'mri','transforms','talairach_with_skull.lta'),
        rawavg = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'mri','rawavg.mgz'),
        lh_pial = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'surf','lh.pial'),
        rh_pial = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'surf','rh.pial'),
        lh_white_h = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'surf','lh.white.H'),
        rh_white_h = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'surf','rh.white.H'),
        lh_white_k = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'surf','lh.white.K'),
        rh_white_k = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'surf','rh.white.K'),
    output:
        touch_fastsurfer=touch(join(config['out_dir'], 'logs', 'sub-' + subject_id + "_fastsurfer_symlinks.done")),
    group: 'preproc'
    threads: 8
    shell:
        "rm {params.talairach_lta} {params.talairach_skull_lta} {params.rawavg} {params.lh_pial} {params.rh_pial} {params.lh_white_h} {params.rh_white_h} {params.lh_white_k} {params.rh_white_k}&&\
        cp {params.talairach_xfm} {params.talairach_lta}&&\
        cp {params.talairach_xfm} {params.talairach_skull_lta}&&\
        cp {input.t1_fname} {params.rawavg}&&\
        cp {params.lh_pial_t1} {params.lh_pial}&&\
        cp {params.rh_pial_t1} {params.rh_pial}&&\
        cp {params.lh_white_preaparc_h} {params.lh_white_h}&&\
        cp {params.rh_white_preaparc_h} {params.rh_white_h}&&\
        cp {params.lh_white_preaparc_k} {params.lh_white_k}&&\
        cp {params.rh_white_preaparc_k} {params.rh_white_k}"

#final_outputs.extend(expand(join(config['out_dir'], 'logs', 'sub-' + subject_id + "_fastsurfer_symlinks.done"), subject=subjects))

rule aparcseg_to_nrrd:
    input:
        segs = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'mri','aparc+aseg.mgz'),
        touch_fastsurfer = rules.fastsurfer_symlinks.output.touch_fastsurfer,
    params:
        atlas_labels = config['fastsurfer_config']['colors'],
        atlas_colors= config['fastsurfer_config']['colors'],
        orien= config['fastsurfer_config']['orien'],
    output:
        seg_nrrd = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'mri','aparc+aseg.seg.nrrd')
    group: 'preproc'
    threads: 8
    script: '../scripts/working/niiTonrrd.py'


if config['seeg_contacts']['present']:
    rule vis_electrodes_native:
        input: 
            fcsv = get_electrodes_coords(subject_id,coords_space='native',coords_type='SEEGA'),
            t1_fname = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'mri','orig.mgz'),
            xfm_noncontrast = get_transform_filename,
            seg_nrrd = rules.aparcseg_to_nrrd.output.seg_nrrd,
        params:
            lh_pial = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'surf','lh.pial'),
            rh_pial = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'surf','rh.pial'),
            lh_sulc = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'surf','lh.sulc'),
            rh_sulc = join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'surf','rh.sulc'),
        output:
            html = report(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='electrodes.html',space='native'),
                    caption='../reports/electrodes_vis.rst',
                    category='Electrodes in template space',
                    subcategory='{desc} reg to {template}'),
        group: 'preproc'
        threads: 8
        script: '../scripts/vis_electrodes_native.py'

    final_outputs.extend(expand(bids(root=join(config['out_dir'], 'derivatives', 'atlasreg'),prefix='sub-'+subject_id+'/qc/sub-'+subject_id,suffix='electrodes.html',space='native',include_subject_dir=False),
                        subject=subjects))

final_outputs.extend(expand(join(config['out_dir'],'derivatives','fastsurfer','sub-' + subject_id, 'mri','aparc+aseg.seg.nrrd'), subject=subjects))
