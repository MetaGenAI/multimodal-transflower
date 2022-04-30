#!/bin/bash

folder=$1
py=python3
n=$(nproc)
#n=6
mpirun="mpirun --use-hwthread-cpus"

#$py ./feature_extraction/process_filenames.py $1 --files_extension npz.annotation.txt --name_processing_function annotation ${@:2}
#find $1 -exec rename -f 's/npz.acts.npy.annotation/annotation/' {} +
#$py feature_extraction/extract_transform2.py $1 --feature_name npz.acts --transforms scaler
#$mpirun $py feature_extraction/apply_transforms.py $@ --feature_name npz.acts --transform_name scaler --new_feature_name acts_scaled
#$py feature_extraction/extract_transform2.py $1 --feature_name npz.obs --transforms scaler
#$mpirun $py feature_extraction/apply_transforms.py $@ --feature_name npz.obs --transform_name scaler --new_feature_name obs_scaled
#$mpirun $py feature_extraction/pad_features.py $@ --files_extension annotation.npy --length 11 --padding_const 72
#./feature_extraction/script_to_list_filenames $1 npz.acts.npy

#$py ./feature_extraction/process_filenames.py $1 --files_extension acts.npy --name_processing_function annotation ${@:2}
#find $1 -exec rename -f 's/npz.acts.npy.annotation/annotation/' {} +
$py feature_extraction/extract_transform2.py $1 --feature_name npz.acts --transforms scaler
$mpirun $py feature_extraction/apply_transforms.py $@ --feature_name npz.acts --transform_name scaler --new_feature_name npz.acts_scaled
$py feature_extraction/extract_transform2.py $1 --feature_name npz.obs_cont --transforms scaler
$mpirun $py feature_extraction/apply_transforms.py $@ --feature_name npz.obs_cont --transform_name scaler --new_feature_name obs_cont_scaled

#$py feature_extraction/extract_transform2.py $1 --feature_name obs_cont_single_nocol --transforms scaler
#$mpirun $py feature_extraction/apply_transforms.py $@ --feature_name obs_cont_single_nocol --transform_name scaler --new_feature_name obs_cont_single_nocol_scaled
#$py feature_extraction/extract_transform2.py $1 --feature_name obs_cont_single_nocol_noarm --transforms scaler
#$mpirun $py feature_extraction/apply_transforms.py $@ --feature_name obs_cont_single_nocol_noarm --transform_name scaler --new_feature_name obs_cont_single_nocol_noarm_scaled
#$py feature_extraction/trim_seqs.py $1 --feature_name obs_cont_single_nocol_noarm --trim_begin 25 --new_feature_name obs_cont_single_nocol_noarm_trim
#$py feature_extraction/extract_transform2.py $1 --feature_name obs_cont_single_nocol_noarm_trim --transforms scaler
#$mpirun $py feature_extraction/apply_transforms.py $@ --feature_name obs_cont_single_nocol_noarm_trim --transform_name scaler --new_feature_name obs_cont_single_nocol_noarm_trim_scaled
#$py feature_extraction/trim_seqs.py $1 --feature_name npz.acts --trim_begin 25 --new_feature_name acts_trim
#$py feature_extraction/extract_transform2.py $1 --feature_name acts_trim --transforms scaler
#$mpirun $py feature_extraction/apply_transforms.py $@ --feature_name acts_trim --transform_name scaler --new_feature_name acts_trim_scaled
#$mpirun $py feature_extraction/pad_features.py $@ --files_extension annotation.npy --length 11 --padding_const 66
