#!/bin/bash

#may need
#export RDMAV_FORK_SAFE=1

folder=$1
#py=python3.8
py=python
#py=python3
n=$(nproc)
#n=1
mpirun="mpirun --use-hwthread-cpus"
#mpirun="mpirun"

#$mpirun -n $n $py feature_extraction/smooth_features.py $@ --feature_name motion_features ##not doing this coz we need to deal with rotations differently, so we're doing it in the edf_motion_utils.py step
echo EXTRACT TRANSFORM MOTION
$py feature_extraction/extract_transform2.py $@ --feature_name motion_features --transforms scaler
echo APPLY TRANSFORM MOTION
$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name motion_features --transform_name scaler --new_feature_name motion_features_scaled1

format=wav
fps=30
$mpirun -n $n $py ./feature_extraction/process_audio.py $@ --audio_format $format --feature_names envelope --fps $fps

echo EXTRACT TRANSFORM AUDIO
$py feature_extraction/extract_transform2.py $@ --feature_name envelope --transforms scaler
echo APPLY TRANSFORM AUDIO
$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name envelope --transform_name scaler --new_feature_name envelope_scaled

feature_extraction/script_to_list_filenames $folder speech.wav_envelope_scaled.npy
feature_extraction/fix_lengths.sh $folder speech.wav_envelope_scaled,motion_features_scaled1
