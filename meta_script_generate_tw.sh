#!/bin/bash
#line=UR5_Tianwei2_obs_act_etc_98_data
#line=UR5_Tianwei_obs_act_etc_2_data
line=UR5_Tianwei_obs_act_etc_1_data
for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss4_tw
do
sbatch slurm_script_generate.slurm $exp $line --data_dir $SCRATCH/data/UR5_processed --use_temperature --temperature 1.0
done

