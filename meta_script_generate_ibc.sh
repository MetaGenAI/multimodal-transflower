#!/bin/bash
#line=UR5_Tianwei2_obs_act_etc_98_data
line=0
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss4_tw3
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss4_tw
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss4_tw_zp
#for exp in transflower_expmap3_tw
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss4_tw_smol
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss4_tw_smol3
#for exp in transflower_expmap3_tw_zp
#for exp in moglow_expmap1_tw
#for exp in transformer_expmap_tw_zp
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss4_tw4
#for exp in transflower_zp_inpdrop_tw
#for exp in transflower_zp_inpdrop3_tw_smol3
#for exp in transflower_zp_inpdrop3_tw_smol4
for exp in transflower_zp_inpdrop3_ibc
do
#sbatch slurm_script_generate.slurm $exp $line --data_dir $SCRATCH/data/UR5_processed --use_temperature --temperature 0.1
sbatch slurm_script_generate.slurm $exp $line --data_dir $SCRATCH/data/ibc_block_push --use_temperature --temperature 1.0 --save_jit
#sbatch slurm_script_generate.slurm $exp $line --data_dir $SCRATCH/data/UR5_processed --zero_pads "npz.obs_scaled,npz.acts_scaled" --save_jit
done

