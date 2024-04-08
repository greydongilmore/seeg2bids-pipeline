

if config['noncontrast_t1']['present']:
    rule import_subj_t1_hippunfold:
        input: 
            in_t1w=get_noncontrast_filename_fs,
        params:
            in_dir = directory(join(config['out_dir'], 'derivatives', 'hippunfold_in')),
            in_dir_sub = directory(join(config['out_dir'], 'derivatives', 'hippunfold_in','sub-'+subject_id)),
        output: 
            out_t1w=bids(root=join(config['out_dir'], 'derivatives','hippunfold_in'), subject=subject_id, datatype='anat', suffix='T1w.nii.gz')
        group: 'preproc'
        shell: "mkdir -p {params.in_dir} && mkdir -p {params.in_dir_sub} && mkdir -p {params.in_dir_sub}/anat && cp {input.in_t1w} {output.out_t1w}"

else:
    rule import_subj_t1_hippunfold:
        input: 
            in_t1w=get_pre_t1_filename,
        params:
            in_dir = directory(join(config['out_dir'], 'derivatives', 'hippunfold_in')),
            in_dir_sub = directory(join(config['out_dir'], 'derivatives', 'hippunfold_in','sub-'+subject_id)),
        output: 
            out_t1w=bids(root=join(config['out_dir'], 'derivatives','hippunfold_in'), subject=subject_id, datatype='anat', suffix='T1w.nii.gz')
        group: 'preproc'
        shell: 
            "mkdir -p {params.in_dir} && mkdir -p {params.in_dir_sub} && mkdir -p {params.in_dir_sub}/anat && cp {input.in_t1w} {output.out_t1w}"

rule hippunfold_seg:
    input: 
        out_t1w=bids(root=join(config['out_dir'], 'derivatives','hippunfold_in'), subject=subject_id, datatype='anat', suffix='T1w.nii.gz')
    params:
        in_dir = directory(join(config['out_dir'], 'derivatives', 'hippunfold_in')),
        hippunfold_out = directory(join(config['out_dir'], 'derivatives','hippunfold')),
        modality = config['hippunfold_config']['modality'],
        path_T1w = bids(root=join(config['out_dir'], 'derivatives','hippunfold_in'), subject=subject_id, datatype='anat', suffix='T1w.nii.gz'),
        participant_label = subject_id,
        hippunfold_container= config['singularity']['hippunfold'],
    output:
        t1_fname = join(config['out_dir'], 'derivatives','hippunfold','hippunfold','sub-' + subject_id, 'anat','sub-' + subject_id + "_desc-preproc_T1w.nii.gz"),
    shell:
        "singularity run -e {params.hippunfold_container} {params.in_dir} {params.hippunfold_out} participant --force_output --participant_label {params.participant_label}\
        --modality {params.modality} --cores 4"

rule hippunfold_post_proc:
    input: 
        t1_fname = join(config['out_dir'], 'derivatives','hippunfold','hippunfold','sub-' + subject_id, 'anat','sub-' + subject_id + "_desc-preproc_T1w.nii.gz"),
    params:
        subject_id = subject_id,
        dseg_labels_file=config['hippunfold_config']['atlas_labels_tsv'],
        deriv_dir = directory(join(config['out_dir'], 'derivatives')),
    output:
        touch_hippunfold=touch(join(config['out_dir'], 'logs', 'sub-' + subject_id + "_hippunfold.done")),
    script:
        "../scripts/working/hippunfold_to_slicer.py"

final_outputs.extend(expand(rules.hippunfold_post_proc.output.touch_hippunfold, subject=subjects))
