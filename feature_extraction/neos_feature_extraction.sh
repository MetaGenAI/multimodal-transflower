
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
#$py feature_extraction/extract_transform2.py $1 --feature_name dat.person1 --transforms scaler
#$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name dat.person1 --transform_name scaler --new_feature_name person1_scaled

#mpi
#./feature_extraction/script_to_list_filenames ${1} dat.person1.npy
#$mpirun -n $n $py ./feature_extraction/split_feats.py $@ --feature_name dat.person1 --split_index 6 --new_feature_names root_scale_pos,rest_feats
#$mpirun -n $n $py ./feature_extraction/split_feats.py $@ --feature_name root_scale_pos --split_index 3 --new_feature_names root_scale,root_pos
#$py feature_extraction/extract_transform2.py $1 --feature_name dat.person1.rel --transforms scaler
#$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name dat.person1.rel --transform_name scaler --new_feature_name rel_feats_scaled
#$py feature_extraction/extract_transform2.py $1 --feature_name root_pos --transforms scaler
#$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name root_pos --transform_name scaler --new_feature_name root_pos_scaled

./feature_extraction/script_to_list_filenames ${1} dat.person1.npy
$mpirun -n $n $py ./feature_extraction/split_feats.py $@ --feature_name dat.person1 --split_index 9 --new_feature_names root_feats1,rest_feats1
$mpirun -n $n $py ./feature_extraction/split_feats.py $@ --feature_name root_feats1 --split_index 3 --new_feature_names root_scale1,root_pos_rot1
$py feature_extraction/extract_transform2.py $1 --feature_name dat.person1.rel --transforms scaler
$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name dat.person1.rel --transform_name scaler --new_feature_name rel_feats_scaled1
$py feature_extraction/extract_transform2.py $1 --feature_name root_pos_rot1 --transforms scaler
$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name root_pos_rot1 --transform_name scaler --new_feature_name root_pos_rot_scaled1
$mpirun -n $n $py ./feature_extraction/split_feats.py $@ --feature_name dat.person2 --split_index 9 --new_feature_names root_feats2,rest_feats2
$mpirun -n $n $py ./feature_extraction/split_feats.py $@ --feature_name root_feats2 --split_index 3 --new_feature_names root_scale2,root_pos_rot2
$py feature_extraction/extract_transform2.py $1 --feature_name dat.person2.rel --transforms scaler
$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name dat.person2.rel --transform_name scaler --new_feature_name rel_feats_scaled2
$py feature_extraction/extract_transform2.py $1 --feature_name root_pos_rot2 --transforms scaler
$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name root_pos_rot2 --transform_name scaler --new_feature_name root_pos_rot_scaled2
$py ./feature_extraction/combine_feats.py $@ $1/base_filenames.txt --feature_names root_pos_rot_scaled1,root_pos_rot_scaled2 --new_feature_name combined_root_pos_rot_scaled
$py ./feature_extraction/combine_feats.py $@ $1/base_filenames.txt --feature_names rel_feats_scaled1,rel_feats_scaled2 --new_feature_name combined_real_feats_scaled

#./feature_extraction/script_to_list_filenames ${1} dat.person1.npy
#$mpirun -n $n $py ./feature_extraction/split_feats.py $@ --feature_name dat.person1 --split_index 6 --new_feature_names root_scale_pos1,rest_feats1
#$mpirun -n $n $py ./feature_extraction/split_feats.py $@ --feature_name root_scale_pos1 --split_index 3 --new_feature_names root_scale1,root_pos1
#$py feature_extraction/extract_transform2.py $1 --feature_name dat.person1.rel --transforms scaler
#$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name dat.person1.rel --transform_name scaler --new_feature_name rel_feats_scaled1
#$py feature_extraction/extract_transform2.py $1 --feature_name root_pos1 --transforms scaler
#$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name root_pos1 --transform_name scaler --new_feature_name root_pos_scaled1
#$mpirun -n $n $py ./feature_extraction/split_feats.py $@ --feature_name dat.person2 --split_index 6 --new_feature_names root_scale_pos2,rest_feats2
#$mpirun -n $n $py ./feature_extraction/split_feats.py $@ --feature_name root_scale_pos2 --split_index 3 --new_feature_names root_scale2,root_pos2
#$py feature_extraction/extract_transform2.py $1 --feature_name dat.person2.rel --transforms scaler
#$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name dat.person2.rel --transform_name scaler --new_feature_name rel_feats_scaled2
#$py feature_extraction/extract_transform2.py $1 --feature_name root_pos2 --transforms scaler
#$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name root_pos2 --transform_name scaler --new_feature_name root_pos_scaled2
#$py ./feature_extraction/combine_feats.py $@ $1/base_filenames.txt --feature_names root_pos_scaled1,root_pos_scaled2 --new_feature_name combined_root_pos_scaled
#$py ./feature_extraction/combine_feats.py $@ $1/base_filenames.txt --feature_names rel_feats_scaled1,rel_feats_scaled2 --new_feature_name combined_real_feats_scaled

######################

#mpi
#./feature_extraction/script_to_list_filenames ${1} dat.person1.npy
#$mpirun -n $n $py ./feature_extraction/split_feats.py $@ --feature_name dat.person1 --split_index 6 --new_feature_names root_scale_pos,rest_feats
#$mpirun -n $n $py ./feature_extraction/split_feats.py $@ --feature_name root_scale_pos --split_index 3 --new_feature_names root_scale,root_pos
##$mpirun -n $n $py ./feature_extraction/extract_deltas.py $@ --feature_name root_pos
##$mpirun -n $n $py ./feature_extraction/combine_feats.py $@ $1/base_filenames.txt --feature_names root_scale,root_pos_deltas,rest_feats --new_feature_name proc_feats
#$py feature_extraction/extract_transform2.py $1 --feature_name dat.person1.rel --transforms scaler
#$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name dat.person1.rel --transform_name scaler --new_feature_name rel_feats_scaled
#$py feature_extraction/extract_transform2.py $1 --feature_name root_pos --transforms scaler
#$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name root_pos --transform_name scaler --new_feature_name root_pos_scaled


#./feature_extraction/script_to_list_filenames ${1} dat.person1.npy
#$mpirun -n $n $py ./feature_extraction/combine_feats.py $@ $1/base_filenames.txt --feature_names dat.person1,dat.person2 --new_feature_name combined_streams
#$py feature_extraction/extract_transform2.py $1 --feature_name combined_streams --transforms scaler
#$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name combined_streams --transform_name scaler --new_feature_name combined_streams_scaled

#srun -n 1 -pty bash -c './feature_extraction/script_to_list_filenames '${1}' dat.person1.npy'
#srun -n $n -pty bash -c $py' ./feature_extraction/combine_feats.py '${1}' '${1}'/base_filenames.txt --feature_names dat.person1,dat.person2 --new_feature_name combined_streams '${@:2}
#srun -n $n -pty bash -c $py' feature_extraction/extract_transform2.py '${1}' --feature_name combined_streams --transforms scaler'
#srun -n $n -pty bash -c $py' feature_extraction/apply_transforms.py '$@' --feature_name combined_streams --transform_name scaler --new_feature_name combined_streams_scaled'
