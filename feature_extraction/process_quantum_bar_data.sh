#!/bin/bash

folder=$1
py=python
#n=$(nproc)
n=6
mpirun="mpirun --use-hwthread-cpus"

$py feature_extraction/extract_transform2.py $1 --feature_name dat.person1.rel --transforms scaler
$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name dat.person1.rel --transform_name scaler --new_feature_name rel_feats_scaled1

format=ogg
fps=30
$mpirun -n $n $py ./feature_extraction/process_audio.py $@ --audio_format $format --feature_names envelope --fps $fps

$py feature_extraction/extract_transform2.py $1 --feature_name envelope --transforms scaler
$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name envelope --transform_name scaler --new_feature_name envelope_scaled
