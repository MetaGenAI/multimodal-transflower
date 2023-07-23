#!/bin/bash

#may need
#export RDMAV_FORK_SAFE=1

folder=$1
#py=python3.8
#py=python
#py=python3
export py=/home/guillefix/miniconda3/bin/python
n=$(nproc)
#n=1
mpirun="mpirun --use-hwthread-cpus"
#mpirun="mpirun"

format=wav
fps=30
$mpirun -n $n $py ./feature_extraction/process_audio.py $@ --audio_format $format --feature_names envelope,mel --mel_feature_size 10 --fps $fps

echo EXTRACT TRANSFORM AUDIO
$py feature_extraction/extract_transform2.py $@ --feature_name envelope_mel  --transforms scaler
echo APPLY TRANSFORM AUDIO
$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name envelope_mel --transform_name scaler --new_feature_name audio_feats_scaled


#abs


#$mpirun -n $n $py feature_extraction/smooth_features.py $@ --feature_name motion_features ##not doing this coz we need to deal with rotations differently, so we're doing it in the edf_motion_utils.py step
#to use with patching of size 3
#$mpirun -n $n $py feature_extraction/pad_features.py data/edf_extracted_data_rel --pad_along_feature_dim --length 21 --feature_name motion_features_abs #TODO: add new_feature_name option here
echo EXTRACT TRANSFORM MOTION
#$py feature_extraction/extract_transform2.py $@ --feature_name motion_features_abs --transforms scaler
#$py feature_extraction/extract_transform2.py $@ --feature_name motion_features_abs_nonsmooth --transforms scaler
#$py feature_extraction/extract_transform2.py $@ --feature_name motion_features_abs_quat --transforms scaler
$py feature_extraction/extract_transform2.py $@ --feature_name motion_features_abs_quat_smoothed --transforms scaler
#$py feature_extraction/extract_transform2.py $@ --feature_name motion_features_abs_expmap_nonsmoothed --transforms scaler
echo APPLY TRANSFORM MOTION
#$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name motion_features_abs --transform_name scaler --new_feature_name motion_features_abs_scaled1
#$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name motion_features_abs_nonsmooth --transform_name scaler --new_feature_name motion_features_abs_nonsmooth_scaled1
#$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name motion_features_abs_quat --transform_name scaler --new_feature_name motion_features_abs_quat_scaled1
$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name motion_features_abs_quat_smoothed --transform_name scaler --new_feature_name motion_features_abs_quat_smoothed_scaled1
#$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name motion_features_abs_expmap_nonsmoothed --transform_name scaler --new_feature_name motion_features_abs_expmap_nonsmoothed_scaled1

#./feature_extraction/fix_lengths.sh $1 speech.wav_envelope_scaled,motion_features_abs_scaled1
#./feature_extraction/fix_lengths.sh $1 speech.wav_envelope_scaled,motion_features_abs_nonsmooth_scaled1
#./feature_extraction/fix_lengths.sh $1 speech.wav_envelope_scaled,motion_features_abs_quat_scaled1
#./feature_extraction/fix_lengths.sh $1 speech.wav_envelope_scaled,motion_features_abs_quat_smoothed_scaled1
./feature_extraction/fix_lengths.sh $1 speech.wav_audio_feats_scaled,motion_features_abs_quat_smoothed_scaled1
#./feature_extraction/fix_lengths.sh $1 speech.wav_audio_feats_scaled,motion_features_abs_expmap_nonsmoothed_scaled1

exit 0

# new audio rep

format=wav
fps=30
$mpirun -n $n $py ./feature_extraction/process_audio.py $@ --audio_format $format --feature_names envelope,mel --mel_feature_size 10 --fps $fps

echo EXTRACT TRANSFORM AUDIO
$py feature_extraction/extract_transform2.py $@ --feature_name envelope_mel  --transforms scaler
echo APPLY TRANSFORM AUDIO
$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name envelope_mel --transform_name scaler --new_feature_name audio_feats_scaled

#./feature_extraction/fix_lengths.sh $1 speech.wav_audio_feats_scaled,motion_features_abs_quat_scaled1
./feature_extraction/fix_lengths.sh $1 speech.wav_audio_feats_scaled,motion_features_abs_quat_smoothed_scaled1

exit 0


#$mpirun -n $n $py feature_extraction/smooth_features.py $@ --feature_name motion_features ##not doing this coz we need to deal with rotations differently, so we're doing it in the edf_motion_utils.py step
#to use with patching of size 3
$mpirun -n $n $py feature_extraction/pad_features.py data/edf_extracted_data_rel --pad_along_feature_dim --length 21 --feature_name motion_features #TODO: add new_feature_name option here
echo EXTRACT TRANSFORM MOTION
$py feature_extraction/extract_transform2.py $@ --feature_name motion_features --transforms scaler
#$py feature_extraction/extract_transform2.py $@ --feature_name motion_features_padded --transforms scaler
echo APPLY TRANSFORM MOTION
#$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name motion_features --transform_name scaler --new_feature_name motion_features_scaled1
$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name motion_features_padded --transform_name scaler --new_feature_name motion_features_scaled1
#$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name motion_features --transform_name scaler --new_feature_name motion_features_scaled_nonpadded1

exit 0


format=wav
fps=30
$mpirun -n $n $py ./feature_extraction/process_audio.py $@ --audio_format $format --feature_names envelope --fps $fps

echo EXTRACT TRANSFORM AUDIO
$py feature_extraction/extract_transform2.py $@ --feature_name envelope --transforms scaler
echo APPLY TRANSFORM AUDIO
$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name envelope --transform_name scaler --new_feature_name envelope_scaled

feature_extraction/script_to_list_filenames $folder speech.wav_envelope_scaled.npy
feature_extraction/fix_lengths.sh $folder speech.wav_envelope_scaled,motion_features_scaled1
