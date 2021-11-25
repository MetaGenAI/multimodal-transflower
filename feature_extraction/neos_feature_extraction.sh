
folder=$1
py=python3
#n=$(nproc)
n=6

./feature_extraction/script_to_list_filenames $1 dat.person1.npy
mpirun -n $n $py ./feature_extraction/combine_feats.py $@ $1/base_filenames.txt --feature_names dat.person1,dat.person2 --new_feature_name combined_streams
mpirun -n 1 $py feature_extraction/extract_transform2.py $1 --feature_name combined_streams --transforms scaler
mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name combined_streams --transform_name scaler --new_feature_name combined_streams_scaled
