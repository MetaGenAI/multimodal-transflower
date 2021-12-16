
folder=$1
py=python
#n=$(nproc)
n=6
mpirun="mpirun --use-hwthread-cpus"

echo $1

#mpirun -n $n $py ./feature_extraction/combine_feats.py $@ $1/base_filenames.txt --feature_names dat.person1,dat.person2 --new_feature_name combined_streams
#mpirun -n 1 $py feature_extraction/extract_transform2.py $1 --feature_name combined_streams --transforms scaler
#mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name combined_streams --transform_name scaler --new_feature_name combined_streams_scaled


#./feature_extraction/script_to_list_filenames ${1} dat.person1.npy
#$py ./feature_extraction/combine_feats.py $@ $1/base_filenames.txt --feature_names dat.person1,dat.person2 --new_feature_name combined_streams
#$py feature_extraction/extract_transform2.py $1 --feature_name combined_streams --transforms scaler
#$py feature_extraction/apply_transforms.py $@ --feature_name combined_streams --transform_name scaler --new_feature_name combined_streams_scaled

#./feature_extraction/script_to_list_filenames ${1} dat.person1.npy
#$py ./feature_extraction/split_feats.py $@ --feature_name dat.person1 --split_index 6 --new_feature_names root_scale_pos,rest_feats
#$py ./feature_extraction/split_feats.py $@ --feature_name root_scale_pos --split_index 3 --new_feature_names root_scale,root_pos
#$py ./feature_extraction/extract_deltas.py $@ --feature_name root_pos
#$py ./feature_extraction/combine_feats.py $@ $1/base_filenames.txt --feature_names root_scale,root_pos_deltas,rest_feats --new_feature_name proc_feats
#$py feature_extraction/extract_transform2.py $1 --feature_name proc_feats --transforms scaler
#$py feature_extraction/apply_transforms.py $@ --feature_name proc_feats --transform_name scaler --new_feature_name proc_feats_scaled
#$py feature_extraction/extract_transform2.py $1 --feature_name root_pos --transforms scaler
#$py feature_extraction/apply_transforms.py $@ --feature_name root_pos --transform_name scaler --new_feature_name root_pos_scaled

#mpi
#./feature_extraction/script_to_list_filenames ${1} dat.person1.npy
#$mpirun -n $n $py ./feature_extraction/split_feats.py $@ --feature_name dat.person1 --split_index 6 --new_feature_names root_scale_pos,rest_feats
#$mpirun -n $n $py ./feature_extraction/split_feats.py $@ --feature_name root_scale_pos --split_index 3 --new_feature_names root_scale,root_pos
#$mpirun -n $n $py ./feature_extraction/extract_deltas.py $@ --feature_name root_pos
#$mpirun -n $n $py ./feature_extraction/combine_feats.py $@ $1/base_filenames.txt --feature_names root_scale,root_pos_deltas,rest_feats --new_feature_name proc_feats
#$py feature_extraction/extract_transform2.py $1 --feature_name proc_feats --transforms scaler
#$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name proc_feats --transform_name scaler --new_feature_name proc_feats_scaled
#$py feature_extraction/extract_transform2.py $1 --feature_name root_pos --transforms scaler
#$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name root_pos --transform_name scaler --new_feature_name root_pos_scaled

./feature_extraction/script_to_list_filenames ${1} dat.person1.npy
$mpirun -n $n $py ./feature_extraction/combine_feats.py $@ $1/base_filenames.txt --feature_names dat.person1,dat.person2 --new_feature_name combined_streams
$py feature_extraction/extract_transform2.py $1 --feature_name combined_streams --transforms scaler
$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name combined_streams --transform_name scaler --new_feature_name combined_streams_scaled

#srun -n 1 -pty bash -c './feature_extraction/script_to_list_filenames '${1}' dat.person1.npy'
#srun -n $n -pty bash -c $py' ./feature_extraction/combine_feats.py '${1}' '${1}'/base_filenames.txt --feature_names dat.person1,dat.person2 --new_feature_name combined_streams '${@:2}
#srun -n $n -pty bash -c $py' feature_extraction/extract_transform2.py '${1}' --feature_name combined_streams --transforms scaler'
#srun -n $n -pty bash -c $py' feature_extraction/apply_transforms.py '$@' --feature_name combined_streams --transform_name scaler --new_feature_name combined_streams_scaled'
