
folder=$1
py=python3
#py=python3.8
#py=python
n=$(nproc)
#n=6

#mpirun -n $n $py feature_extraction/fix_lengths.py $1 $1/base_filenames.txt --fix_length_types none,end,end --modalities dance_style,expmap_cr_scaled_20,audio_feats_scaled_20
#$py feature_extraction/fix_lengths.py $1 $1/base_filenames.txt --fix_length_types none,end,end --modalities dance_style,expmap_cr_scaled_20,audio_feats_scaled_20
#$py feature_extraction/fix_lengths.py $1 $1/base_filenames.txt --fix_length_types end,end --modalities expmap_cr_scaled_60,audio_feats_scaled_60
#$py feature_extraction/fix_lengths.py $1 $1/base_filenames.txt --fix_length_types end,end --modalities expmap_scaled_20,audio_feats_scaled_20
#mpirun -n $n $py feature_extraction/fix_lengths.py $1 $1/base_filenames.txt --fix_length_types end,end --modalities expmap_scaled_20,audio_feats_scaled_20
#mpirun -n $n $py feature_extraction/fix_lengths.py $1 $1/base_filenames.txt --fix_length_types end,end --modalities speech.wav_envelope_scaled,motion_features_scaled1
#$py feature_extraction/fix_lengths.py $1 $1/base_filenames_train.txt --fix_length_types end,end --modalities expmap_scaled_20,audio_feats_scaled_20
#$py feature_extraction/fix_lengths.py $1 $1/base_filenames.txt --fix_length_types end,end --modalities ogg_envelope_scaled,rel_feats_scaled1
#mpirun -n $n $py feature_extraction/fix_lengths.py $1 $1/base_filenames.txt --fix_length_types end,end --modalities ${@:2}
mpirun -n $n $py feature_extraction/fix_lengths.py $1 $1/base_filenames_train.txt --fix_length_types end,end --modalities ${@:2}
