#!/bin/bash

folder=$1
py=python3
#n=$(nproc)
n=6
#mpirun="mpirun --use-hwthread-cpus"
mpirun=""

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

#THIS
#$py feature_extraction/extract_transform2.py $1 --feature_name npz.acts --transforms scaler
#$mpirun $py feature_extraction/apply_transforms.py $@ --feature_name npz.acts --transform_name scaler --new_feature_name npz.acts_scaled
#$py feature_extraction/extract_transform2.py $1 --feature_name npz.obs_cont --transforms scaler
#$mpirun $py feature_extraction/apply_transforms.py $@ --feature_name npz.obs_cont --transform_name scaler --new_feature_name obs_cont_scaled
#

#$py feature_extraction/extract_transform2.py $1 --feature_name obs_cont_single_nocol --transforms scaler
#$mpirun $py feature_extraction/apply_transforms.py $@ --feature_name obs_cont_single_nocol --transform_name scaler --new_feature_name obs_cont_single_nocol_scaled

#THIS
#$py feature_extraction/extract_transform2.py $1 --feature_name obs_cont_single_nocol_noarm --transforms scaler
#$mpirun $py feature_extraction/apply_transforms.py $@ --feature_name obs_cont_single_nocol_noarm --transform_name scaler --new_feature_name obs_cont_single_nocol_noarm_scaled

#echo hi
#$py -c 'print("Hi")'

#LATEST
$mpirun $py feature_extraction/filter_out_by_length.py $1 --feature_name obs_cont_single_nocol_noarm --min_length 50
if [ "$OMPI_COMM_WORLD_RANK" = "0" ]
then
	./feature_extraction/script_to_list_filenames $1 obs_cont_single_nocol_noarm.npy
fi
$py feature_extraction/trim_seqs.py $1 --feature_name obs_cont_single_nocol_noarm --trim_begin 25 --new_feature_name obs_cont_single_nocol_noarm_trim
$py feature_extraction/extract_transform2.py $1 --feature_name obs_cont_single_nocol_noarm_trim --transforms scaler
$mpirun $py feature_extraction/apply_transforms.py $@ --feature_name obs_cont_single_nocol_noarm_trim --transform_name scaler --new_feature_name obs_cont_single_nocol_noarm_trim_scaled
#
$mpirun $py ./feature_extraction/combine_feats.py $@ $1/base_filenames.txt --feature_names npz.times_to_go,npz.times_to_go --new_feature_name duplicated_times_to_go
#
$py feature_extraction/extract_transform2.py $1 --feature_name npz.times_to_go --transforms scaler
$mpirun $py feature_extraction/apply_transforms.py $@ --feature_name npz.times_to_go --transform_name scaler --new_feature_name times_to_go_scaled
$py feature_extraction/extract_transform2.py $1 --feature_name duplicated_times_to_go --transforms scaler
$mpirun $py feature_extraction/apply_transforms.py $@ --feature_name duplicated_times_to_go --transform_name scaler --new_feature_name duplicated_times_to_go_scaled
#
$mpirun $py feature_extraction/trim_seqs.py $@ --feature_name times_to_go_scaled --trim_begin 25 --new_feature_name times_to_go_scaled_trim
$mpirun $py feature_extraction/trim_seqs.py $@ --feature_name duplicated_times_to_go_scaled --trim_begin 25 --new_feature_name duplicated_times_to_go_scaled_trim

$py feature_extraction/trim_seqs.py $1 --feature_name npz.acts --trim_begin 25 --new_feature_name acts_trim
$py feature_extraction/extract_transform2.py $1 --feature_name acts_trim --transforms scaler
$mpirun $py feature_extraction/apply_transforms.py $@ --feature_name acts_trim --transform_name scaler --new_feature_name acts_trim_scaled

#$py feature_extraction/extract_transform2.py $@ --feature_name obs_cont_single_nocol_noarm_incsize_trim --transforms scaler
#$mpirun $py feature_extraction/apply_transforms.py $@ --feature_name obs_cont_single_nocol_noarm_incsize_trim --transform_name scaler --new_feature_name obs_cont_single_nocol_noarm_incsize_trim_scaled

#$py feature_extraction/trim_seqs.py $1 --feature_name obs_cont_single_nocol_noarm --trim_begin 10 --new_feature_name obs_cont_single_nocol_noarm_trim
#$py feature_extraction/extract_transform2.py $1 --feature_name obs_cont_single_nocol_noarm_incsize --transforms scaler
#$mpirun $py feature_extraction/apply_transforms.py $@ --feature_name obs_cont_single_nocol_noarm_incsize --transform_name scaler --new_feature_name obs_cont_single_nocol_noarm_incsize_scaled

#$mpirun $py feature_extraction/trim_seqs.py $@ --feature_name obs_cont_single_nocol_noarm_incsize --trim_begin 25 --new_feature_name obs_cont_single_nocol_noarm_incsize_trim
#$py feature_extraction/extract_transform2.py $@ --feature_name obs_cont_single_nocol_noarm_incsize_trim --transforms scaler
#$mpirun $py feature_extraction/apply_transforms.py $@ --feature_name obs_cont_single_nocol_noarm_incsize_trim --transform_name scaler --new_feature_name obs_cont_single_nocol_noarm_incsize_trim_scaled


#$py feature_extraction/trim_seqs.py $1 --feature_name npz.times_to_go --trim_begin 10 --new_feature_name times_to_go_trim
#$py feature_extraction/extract_transform2.py $1 --feature_name times_to_go_trim --transforms scaler
#$mpirun $py feature_extraction/apply_transforms.py $@ --feature_name times_to_go_trim --transform_name scaler --new_feature_name times_to_go_trim_scaled
#$mpirun $py feature_extraction/pad_features.py $@ --files_extension annotation.npy --length 11 --padding_const 66
