#!/bin/bash
#line=UR5_Tianwei2_obs_act_etc_98_data
#line=UR5_Tianwei_obs_act_etc_4_data
#line=UR5_Tianwei6_obs_act_etc_53_data
line=UR5_Guillermo_obs_act_etc_8_data
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
#for exp in transflower_zp_inpdrop3_tw
#for exp in transflower_zp4_tw
#for exp in transflower_zp5_tw
#for exp in transflower_zp5_short2_tw_smol4
#for exp in transflower_zp5_short2_tw
#for exp in transflower_zp5_short2_tw_paint
#for exp in transflower_zp5_short3_tw
#for exp in transflower_zp5_short3_tw_smol4
#for exp in transflower_zp5_short3_tw_paint
#for exp in transflower_zp5_short4_tw
#for exp in transflower_zp5_short4_tw_smol4
#for exp in transflower_zp5_short2_tw_paint
#for exp in transflower_zp5_short_single_obj_tw_single
#for exp in transflower_zp5_short_single_obj_tw_single_smol4
#for exp in transflower_zp5_short4_tw
#for exp in transflower_zp5_short4_tw
#for exp in transflower_zp5_single_obj_tw_single_smol4
#for exp in transflower_zp5_single_obj_tw_single transflower_zp5_short_single_obj_tw_single transflower_zp5_short_single_obj_nocol_tw_single
#for exp in transflower_zp5_short_single_obj_nocol_tw_single
#for exp in transflower_zp5_short_single_obj_nocol_tw_single_smol4
#for exp in transflower_zp5_short_single_obj_nocol_dp_tw_single_filtered
for exp in transflower_zp5_single_obj_nocol_trim_tw_single_filtered
do
#sbatch slurm_script_generate.slurm $exp $line --data_dir $SCRATCH/data/UR5_processed --use_temperature --temperature 0.1
sbatch slurm_script_generate.slurm $exp $line --data_dir $SCRATCH/data/UR5_processed --use_temperature --zero_pads "obs_cont_scaled,npz.acts_scaled" --temperature 1.0 --save_jit
#sbatch slurm_script_generate.slurm $exp $line --data_dir $SCRATCH/data/UR5_processed --zero_pads "npz.obs_scaled,npz.acts_scaled" --save_jit
done

