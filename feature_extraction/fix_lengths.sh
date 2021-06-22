
folder=$1
py=python3
#n=$(nproc)
n=6

mpirun -n $n $py feature_extraction/fix_lengths.py $1 $1/base_filenames.txt --fix_length_types none,end,end --modalities dance_style,expmap_cr_scaled_20,audio_feats_scaled_20
