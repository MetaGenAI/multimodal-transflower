#!/bin/bash

folder=$1
py=python3
n=$(nproc)
#n=6
mpirun="mpirun --use-hwthread-cpus"

$py feature_extraction/extract_transform2.py $1 --feature_name obs --transforms scaler
$mpirun $py feature_extraction/apply_transforms.py $@ --feature_name obs --transform_name scaler --new_feature_name obs_scaled
$py feature_extraction/extract_transform2.py $1 --feature_name acts --transforms scaler
$mpirun $py feature_extraction/apply_transforms.py $@ --feature_name acts --transform_name scaler --new_feature_name acts_scaled
