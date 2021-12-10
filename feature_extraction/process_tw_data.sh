#!/bin/bash

folder=$1
py=python3
n=$(nproc)
#n=6

#$py ./feature_extraction/process_filenames.py $1 --files_extension acts.npy --name_processing_function annotation ${@:2}
#find $1 -exec rename 's/npz.acts.npy.annotation/annotation/' {} +
#$py feature_extraction/extract_transform2.py $1 --feature_name acts --transforms scaler
#$py feature_extraction/apply_transforms.py $@ --feature_name acts --transform_name scaler --new_feature_name acts_scaled
#$py feature_extraction/extract_transform2.py $1 --feature_name obs --transforms scaler
#$py feature_extraction/apply_transforms.py $@ --feature_name obs --transform_name scaler --new_feature_name obs_scaled
$py feature_extraction/pad_features.py $@ --files_extension annotation.npy --length 11 --padding_const 66
