
folder=$1
py=python3
# n=$(nproc)
n=8
# mpirun="mpirun --use-hwthread-cpus"
mpirun="mpirun"

#target fps
fps=30

# code for Expmap representations from bvhs
param=expmap
#param=position

# motion
$mpirun -n $n $py feature_extraction/process_motions.py $@ --param ${param} --fps $fps #--do_mirror #mirror not working with this data atm
$mpirun -n 1 $py feature_extraction/extract_transform2.py $1 --feature_name bvh_${param} --transforms scaler
$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name bvh_${param} --transform_name scaler --new_feature_name ${param}_scaled_${fps}
cp $1/motion_expmap_data_pipe.sav $1/motion_${param}_scaled_${fps}_data_pipe.sav

# #### audio

format=wav
$mpirun -n $n $py ./feature_extraction/process_audio.py $@ --audio_format $format --feature_names envelope,mel --mel_feature_size 10 --fps $fps

echo EXTRACT TRANSFORM AUDIO
$py feature_extraction/extract_transform2.py $@ --feature_name envelope_mel  --transforms scaler
echo APPLY TRANSFORM AUDIO
$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name envelope_mel --transform_name scaler --new_feature_name audio_feats_scaled

./feature_extraction/script_to_list_filenames $folder wav_audio_feats_scaled.npy
./feature_extraction/fix_lengths.sh $folder wav_audio_feats_scaled,expmap_scaled_30

#audio feats 11
#motion feats 255