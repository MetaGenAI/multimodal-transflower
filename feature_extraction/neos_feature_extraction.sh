
folder=$1
py=python
#n=$(nproc)
#n=6

echo $1

#mpirun -n $n $py ./feature_extraction/combine_feats.py $@ $1/base_filenames.txt --feature_names dat.person1,dat.person2 --new_feature_name combined_streams
#mpirun -n 1 $py feature_extraction/extract_transform2.py $1 --feature_name combined_streams --transforms scaler
#mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name combined_streams --transform_name scaler --new_feature_name combined_streams_scaled


./feature_extraction/script_to_list_filenames ${1} dat.person1.npy
$py ./feature_extraction/combine_feats.py $@ $1/base_filenames.txt --feature_names dat.person1,dat.person2 --new_feature_name combined_streams
$py feature_extraction/extract_transform2.py $1 --feature_name combined_streams --transforms scaler
$py feature_extraction/apply_transforms.py $@ --feature_name combined_streams --transform_name scaler --new_feature_name combined_streams_scaled

#./feature_extraction/script_to_list_filenames ${1} dat.person1.npy
#$py feature_extraction/extract_transform2.py $1 --feature_name dat.person1 --transforms scaler
#$py feature_extraction/apply_transforms.py $@ --feature_name dat.person1 --transform_name scaler --new_feature_name person1_scaled

#srun -n 1 -pty bash -c './feature_extraction/script_to_list_filenames '${1}' dat.person1.npy'
#srun -n $n -pty bash -c $py' ./feature_extraction/combine_feats.py '${1}' '${1}'/base_filenames.txt --feature_names dat.person1,dat.person2 --new_feature_name combined_streams '${@:2}
#srun -n $n -pty bash -c $py' feature_extraction/extract_transform2.py '${1}' --feature_name combined_streams --transforms scaler'
#srun -n $n -pty bash -c $py' feature_extraction/apply_transforms.py '$@' --feature_name combined_streams --transform_name scaler --new_feature_name combined_streams_scaled'
