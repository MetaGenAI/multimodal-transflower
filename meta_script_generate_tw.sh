#!/bin/bash
#line=UR5_Tianwei2_obs_act_etc_98_data
line=UR5_Tianwei_obs_act_etc_4_data
for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss4_tw3
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss4_tw4
do
#sbatch slurm_script_generate.slurm $exp $line --data_dir $SCRATCH/data/UR5_processed --use_temperature --temperature 0.1
sbatch slurm_script_generate.slurm $exp $line --data_dir $SCRATCH/data/UR5_processed --use_temperature --zero_pads "npz.obs_scaled,npz.acts_scaled" --temperature 0.1 --save_jit
done

