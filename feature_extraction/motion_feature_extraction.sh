
folder=$1
py=python3
#py=/gpfslocalsup/pub/anaconda-py3/2020.02/envs/pytorch-cpu-1.7.1/bin/python
n=$(nproc)
#n=40
mpirun="mpirun --use-hwthread-cpus"

#target fps
fps=20

# to convert aistpp to BVH with mixamo skeleton
#mpirun -n $n $py feature_extraction/process_aistpp.py $@ --fps 60 # this fps is the source fps of aistpp which is 60Hz

# code for Expmap representations from bvhs
param=expmap
#param=position

#nompi
#srun -n $n -pty bash -c $py' ./feature_extraction/process_motions.py '$@' --param '${param}' --fps '$fps' --do_mirror'
#srun -n 1 -pty $py feature_extraction/extract_transform2.py $1 --feature_name bvh_${param} --transforms scaler
#srun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name bvh_${param} --transform_name scaler --new_feature_name ${param}_scaled_${fps}
#cp $1/motion_expmap_data_pipe.sav $1/motion_${param}_scaled_${fps}_data_pipe.sav

#new
$mpirun -n $n $py feature_extraction/process_motions.py $@ --param ${param} --fps $fps --do_mirror
$mpirun -n 1 $py feature_extraction/extract_transform2.py $1 --feature_name bvh_${param} --transforms scaler
$mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name bvh_${param} --transform_name scaler --new_feature_name ${param}_scaled_${fps}
cp $1/motion_expmap_data_pipe.sav $1/motion_${param}_scaled_${fps}_data_pipe.sav

#with smoothing
# $mpirun -n $n $py feature_extraction/process_motions.py $@ --param ${param} --fps $fps --joint_rotation_smoothing 0.5 --feature_name ${param}_smoothed
# $mpirun -n 1 $py feature_extraction/extract_transform2.py $1 --feature_name bvh_${param}_smoothed --transforms scaler
# $mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name bvh_${param}_smoothed --transform_name scaler --new_feature_name ${param}_smoothed_scaled_${fps}
# cp $1/motion_expmap_data_pipe.sav $1/motion_${param}_smoothed_scaled_${fps}_data_pipe.sav

#with constant remover (old)
#mpirun -n $n $py feature_extraction/process_motions.py $@ --param ${param} --fps $fps --do_mirror
#rename -f 's/bvh_expmap/bvh_expmap_cr/' $1/*bvh_expmap.npy
#mpirun -n 1 $py feature_extraction/extract_transform2.py $1 --feature_name bvh_${param}_cr --transforms scaler
#mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name bvh_${param}_cr --transform_name scaler --new_feature_name ${param}_cr_scaled_${fps}
#cp $1/motion_expmap_data_pipe.sav $1/motion_${param}_cr_scaled_${fps}_data_pipe.sav

#if doing mirroring
#feature_extraction/duplicate_features.sh $1 audio_feats_scaled_20

##################other

# code for Rotmat representation for AISTPP
#mpirun -n $n $py ./scripts/feature_extraction/aistpp_to_rotmats.py $@
#mpirun -n 1 $py ./scripts/feature_extraction/extract_transform2.py $@ --feature_name pkl_joint_angles_mats --transforms scaler
# mpirun -n $n $py ./scripts/feature_extraction/apply_transforms.py $@ --feature_name pkl_joint_angles_mats --transform_name scaler --new_feature_name joint_angles_scaled


#no mpi
#$py feature_extraction/process_motions.py $@ --param ${param} --fps $fps --do_mirror
#rename 's/bvh_expmap/bvh_expmap_cr/' $1/*bvh_expmap.npy
#mpirun -n 1 $py feature_extraction/extract_transform2.py $1 --feature_name bvh_${param}_cr --transforms scaler
#$py feature_extraction/apply_transforms.py $@ --feature_name bvh_${param}_cr --transform_name scaler --new_feature_name ${param}_cr_scaled_${fps}
#cp $1/motion_expmap_data_pipe.sav $1/motion_${param}_cr_scaled_${fps}_data_pipe.sav


# for moglow
#param=position
#mpirun -n 1 $py feature_extraction/extract_transform2.py $1 --feature_name moglow_loc --transforms scaler
#mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name moglow_loc --transform_name scaler --new_feature_name ${param}_scaled
#cp $1/moglow_loc_scaler.pkl $1/moglow_position_scaled_scaler.pkl
#mpirun -n 1 $py feature_extraction/extract_transform2.py $1 --feature_name moglow_loc_control --transforms scaler
#mpirun -n $n $py feature_extraction/apply_transforms.py $@ --feature_name moglow_loc_control --transform_name scaler --new_feature_name moglow_control_scaled
#cp $1/moglow_loc_control_scaler.pkl $1/moglow_control_scaled_scaler.pkl

