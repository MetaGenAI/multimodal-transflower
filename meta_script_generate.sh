#!/bin/bash

#exp=transglower_moglow_pos
#exp=transglower_residual_moglow_pos
#exp=transflower_residual_moglow_pos
#exp=transflower_moglow_pos
#exp=residualflower2_transflower_moglow_pos
#exp=moglow_moglow_pos

#exp=transglower_aistpp_expmap
#exp=transglower_residual_aistpp_expmap
#exp=transflower_residual_aistpp_expmap
#exp=transflower_aistpp_expmap
#exp=residualflower2_transflower_aistpp_expmap
#exp=moglow_aistpp_expmap

#base_filenames_file=base_filenames_test.txt
#base_filenames_file=base_filenames_test_test.txt
#base_filenames_file=base_filenames_aistpp_test.txt
#base_filenames_file=base_filenames_aistpp_train_sample.txt
#base_filenames_file=base_filenames_test_test2.txt
base_filenames_file=base_filenames_test2.txt
#line=data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_2_ID2C00_streams
#line=data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_2_ID2C00_streams
line=data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams
#line=data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_2_ID2C00_streams
#exp=transflower_expmap_cr4_bs5_og2_futureN_gauss4_neosraw2
#exp=moglow_expmap1_tf2_neosraw2
#exp=moglow_expmap1_tf3_single_kulzaworld_neosraw
#exp=transflower_expmap_cr4_bs5_og2_futureN_gauss5_single_kulzaworld_neosraw
#exp=transflower_expmap_cr4_bs5_og2_futureN_gauss4_single_kulzaworld_neosraw
#exp=transflower_expmap_cr4_bs5_og2_futureN_gauss5_single2_kulzaworld_neosraw
#for exp in moglow_expmap1_tf2_single_kulzaworld_neosraw moglow_expmap1_tf3_single_kulzaworld_neosraw transflower_expmap_cr4_bs5_og2_futureN_gauss5_single2_kulzaworld_neosraw
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single2_kulzaworld_neosraw_rel
#for exp in moglow_expmap1_tf3_rel_single_kulzaworld_neosraw_rel
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single2_kulzaworld_neosraw_rel_nonshuff moglow_expmap1_tf3_rel_single_kulzaworld_neosraw_rel
#for exp in discrete_model_kulzaworld_neosraw_rel2
#for exp in discrete_model_kulzaworld_neosraw_rel
#for exp in discrete_model2_kulzaworld_neosraw_rel2
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single3_kulzaworld_neosraw_rel_smol
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single2_kulzaworld_neosraw_rel2
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_combined_dekaworld_neosraw_rel
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_combined2_dekaworld_neosraw_rel
for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single4_kulzaworld_neosraw_rel2
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss5_single3_kulzaworld_neosraw_fixed
#for exp in moglow_expmap1_tf2_single_kulzaworld_neosraw

do
#exp=transflower_expmap_cr4_bs5_og2_futureN_gauss4_neosraw4
#exp=transflower_expmap_cr4_bs5_og2_futureN_gauss4_single_kulzaworld_neosraw
#exp=moglow_expmap1_tf2_single_kulzaworld_neosraw
#sbatch slurm_script_generate.slurm $exp $line --data_dir $SCRATCH/data/dekaworld_alex_guille_neosdata --seeds "combined_streams_scaled,data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_1_ID1E66900_streams" --max_length 1024
#sbatch slurm_script_generate.slurm $exp $line --data_dir $SCRATCH/data/dekaworld_alex_guille_neosdata --zero_seeds "combined_streams_scaled" --sequence_length 1024
#sbatch slurm_script_generate.slurm $exp $line --data_dir data/dekaworld_alex_guille_neosdata2 --seeds "combined_streams_scaled,data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_2_ID2C00_streams" --sequence_length 2048 --max_length 120 #--generate_ground_truth
#sbatch slurm_script_generate.slurm $exp $line --data_dir $SCRATCH/data/dekaworld_alex_guille_neosdata3 --seeds "combined_streams_scaled,data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_2_ID2C00_streams" --sequence_length 2048 --max_length 120 #--generate_ground_truth
#sbatch slurm_script_generate.slurm $exp $line --data_dir $SCRATCH/data/kulzaworld_guille_neosdata_npy --seeds "person1_scaled,data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_2_ID2C00_streams" --sequence_length 2048 --max_length 120 #--generate_ground_truth
#sbatch slurm_script_generate.slurm $exp $line --data_dir $SCRATCH/data/kulzaworld_guille_neosdata_npy --sequence_length 2000 --max_length 120 #--generate_ground_truth

#sbatch slurm_script_generate.slurm $exp $line --data_dir $SCRATCH/data/kulzaworld_guille_neosdata_npy2 --sequence_length 2048 --seeds "root_pos_scaled,data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_2_ID2C00_streams" --zero_seeds "feats_proc_scaled" #--generate_ground_truth
#sbatch slurm_script_generate.slurm $exp $line --data_dir $SCRATCH/data/kulzaworld_guille_neosdata_npy_relative --sequence_length 2048 --seeds "root_pos_scaled,data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams"
#sbatch slurm_script_generate.slurm $exp $line --data_dir $SCRATCH/data/kulzaworld_guille_neosdata_npy_relative --sequence_length 2048 #--teacher_forcing
sbatch slurm_script_generate.slurm $exp $line --data_dir $SCRATCH/data/kulzaworld_guille_neosdata_npy_relative --sequence_length 2048 --use_temperature --temperature 0.1 --save_jit
#sbatch slurm_script_generate.slurm $exp $line --data_dir $SCRATCH/data/dekaworld_alex_guille_neosdata_npy_relative --sequence_length 2048 --use_temperature --temperature 0.1 --save_jit
#sbatch slurm_script_generate.slurm $exp $line --data_dir $SCRATCH/data/kulzaworld_guille_neosdata_npy_relative --sequence_length 2048 --use_temperature --temperature 0.5
#sbatch slurm_script_generate.slurm $exp $line --data_dir $SCRATCH/data/kulzaworld_guille_neosdata_npy2 --sequence_length 2048 
#
#sbatch slurm_script_generate.slurm $exp $line --data_dir $SCRATCH/data/dekaworld_alex_guille_neosdata --seeds "combined_streams_scaled,data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_1_ID1E66900_streams" --sequence_length 1024 --max_length 1144 #--generate_ground_truth
done

#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss4_neosraw
#
#do
#	while read line; do
#		  echo "$line"
#		#sbatch slurm_script_generate.slurm $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined3 --seeds "expmap_cr_scaled_20,kthmisc_gCA_sFM_cAll_d01_mCA_ch14" --generate_video --max_length 1024 --zero_seeds "expmap_cr_scaled_20" #--generate_ground_truth
#		#sbatch slurm_script_generate.slurm $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined3 --seeds "dance_style,justdance_gJD_sFM_cAll_d01_mCA31_ch31" --generate_video --zero_seeds "expmap_cr_scaled_20" #--max_length 1024 
#		#sbatch slurm_script_generate.slurm $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined3 --seeds "expmap_cr_scaled_20,kthmisc_gCA_sFM_cAll_d01_mCA_ch14;dance_style,kthmisc_gCA_sFM_cAll_d01_mCA_ch14" --generate_video --max_length 1024 --audio_format wav --generate_ground_truth
#		#sbatch slurm_script_generate.slurm $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined3 --generate_video --max_length 1024 --audio_format wav --generate_ground_truth
#		#sbatch slurm_script_generate.slurm $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined3 --generate_video --max_length 1024 --audio_format wav
#
#		#for i in 1 2 3 4 5; do
#		#	mkdir inference/generated_${i}/
#		#	mkdir inference/generated_${i}/${exp}
#		#	mkdir inference/generated_${i}/${exp}/predicted_mods
#		#	mkdir inference/generated_${i}/${exp}/videos
#		#	#sbatch slurm_script_generate.slurm $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined_test${i} --output_folder=inference/generated_${i}
#		#	sbatch slurm_script_generate.slurm $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined_test${i} --output_folder=inference/generated_${i} --seeds_file $SCRATCH/data/seeds_${i}
#		#done
#	done <$base_filenames_file
#
#	#for i in 1 2 3 4 5; do
#	#for i in 5; do
#	#	mkdir inference/generated_${i}/
#	#	mkdir inference/generated_${i}/${exp}
#	#	mkdir inference/generated_${i}/${exp}/predicted_mods
#	#	mkdir inference/generated_${i}/${exp}/videos
#	#	#sbatch slurm_script_generate.slurm $exp $line --generate_bvh --data_dir $SCRATCH/data/dance_combined_test${i} --output_folder=inference/generated_${i}
#	#	#sbatch slurm_script_generate_array.slurm $exp $i
#	#	sbatch slurm_script_generate_array_rs.slurm $exp $i
#	#done
#
#	#sbatch slurm_script_generate.slurm $exp aistpp_gMH_sFM_cAll_d22_mMH3_ch04 --generate_bvh --data_dir=${SCRATCH}/data/dance_combined2
#	#sbatch slurm_script_generate.slurm $exp fan
#	#sbatch slurm_script_generate.slurm $exp polish_cow
#	#sbatch slurm_script_generate.slurm $exp aistpp_gLO_sBM_cAll_d14_mLO4_ch02
#done

#for exp in transflower_residual_aistpp_expmap_future1 transflower_aistpp_expmap_future1 residualflower2_transflower_aistpp_expmap_future1_future1
#do
#	sbatch slurm_script_generate.slurm $exp
#done
