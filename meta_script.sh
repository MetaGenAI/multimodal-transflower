#!/bin/bash

module purge
module load pytorch-gpu/py3/1.8.0

#for exp in moglow_expmap1
#for exp in moglow_expmap1_tf
#for exp in moglow_expmap1_label
#for exp in moglow_expmap1_label2
#for exp in moglow_expmap1_label3
#for exp in moglow_expmap1_label4
#for exp in moglow_expmap1_label3b
#for exp in moglow_expmap1_label3c
#for exp in moglow_expmap1_label4b
#for exp in moglow_expmap1_label4c
#for exp in moglow_expmap1_label4d
#for exp in moglow_expmap1_label4e
#for exp in transflower_expmap_cr4_label_bs5c
#for exp in transflower_expmap_cr4_label_bs5 transflower_expmap_cr
#for exp in transflower_expmap_cr4_label_bs5
#for exp in transflower_expmap_cr4_label_bs5_og
#for exp in transflower_expmap_cr4_bs5_og transflower_expmap_cr4_label_bs5_og
#for exp in transflower_expmap_cr4_bs5_og2_futureN
#for exp in transflower_expmap_cr4_bs5_og_futureN
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss2
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss3
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss4
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss4_single
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss4_single2 transflower_expmap_cr4_bs5_og2_futureN_gauss5_single2
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss5_single
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss5_single2
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss5_single3
for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single2
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single3
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single4
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss4 transflower_expmap_cr4_bs5_og2_futureN_gauss5
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss5
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss6
#for exp in moglow_expmap1_tf2
#for exp in moglow_expmap1_tf2_single
#for exp in moglow_expmap1_tf2_single moglow_expmap1_tf3_single
#for exp in moglow_expmap1_tf3_single
#for exp in moglow_expmap1_tf3_rel_single
#for exp in discrete_model
#for exp in discrete_model2

#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_combined
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_combined2
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_combined2_both

do
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_aistpp --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered3.txt --learning_rate 1e-4 --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_aistppkth --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered8.txt --learning_rate 1e-4 --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --learning_rate 1e-4 --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_lr2_sm --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered9.txt --learning_rate 5e-4 --continue_train
	
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_neosraw --num_nodes 1 --max_epochs 3000 --data_dir=$SCRATCH/data/dekaworld_alex_guille_neosdata --base_filenames_file base_filenames.txt --learning_rate 5e-4 --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_neosraw2 --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/dekaworld_alex_guille_neosdata2 --base_filenames_file base_filenames.txt --learning_rate 5e-4 --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_neosraw4 --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/dekaworld_alex_guille_neosdata3 --base_filenames_file base_filenames.txt --learning_rate 5e-4 --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_kulzaworld_neosraw --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/kulzaworld_guille_neosdata_npy --base_filenames_file base_filenames.txt --learning_rate 1e-6 --continue_train 
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_kulzaworld_neosraw --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/kulzaworld_guille_neosdata_npy --base_filenames_file base_filenames.txt --learning_rate 1e-6 --continue_train --no_load_hparams --load_weights_only --load_optimizer_states --lr_policy none
	#sbatch slurm4s.slurm $exp --experiment_name ${exp}_kulzaworld_neosraw --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/kulzaworld_guille_neosdata_npy2 --base_filenames_file base_filenames.txt --learning_rate 1e-6 --load_weights_only --gradient_clip_val 10.0 --continue_train --no_load_hparams --load_optimizer_states --lr_policy none --batch_size 48
#	sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_kulzaworld_neosraw --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/kulzaworld_guille_neosdata_npy2 --base_filenames_file base_filenames.txt --learning_rate 1e-5
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_kulzaworld_neosraw --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/kulzaworld_guille_neosdata_npy2 --base_filenames_file base_filenames.txt --learning_rate 1e-5 --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_kulzaworld_neosraw_rel --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/kulzaworld_guille_neosdata_npy_relative --base_filenames_file base_filenames.txt --learning_rate 1e-5 --continue_train
	#sbatch slurm_script1.slurm $exp --experiment_name ${exp}_kulzaworld_neosraw_rel_smolbatch --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/kulzaworld_guille_neosdata_npy_relative --base_filenames_file base_filenames.txt --learning_rate 1e-5
	#sbatch slurm_script4.slurm $exp --experiment_name ${exp}_kulzaworld_neosraw_fixed --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/kulzaworld_guille_neosdata_npy2 --base_filenames_file base_filenames.txt --continue_train


	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_neosraw4 --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/dekaworld_alex_guille_neosdata3 --base_filenames_file base_filenames.txt --gradient_clip_val 10.0 --learning_rate 1e-5

	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_kulzaworld_neosraw_rel --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/kulzaworld_guille_neosdata_npy_relative --base_filenames_file base_filenames.txt --learning_rate 1e-4
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_kulzaworld_neosraw_rel_smol --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/kulzaworld_guille_neosdata_npy_relative --base_filenames_file base_filenames_train2.txt --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_kulzaworld_neosraw_rel2 --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/kulzaworld_guille_neosdata_npy_relative --base_filenames_file base_filenames.txt
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_kulzaworld_neosraw_rel2 --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/kulzaworld_guille_neosdata_npy_relative --base_filenames_file base_filenames.txt --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_kulzaworld_neosraw_rel2 --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/kulzaworld_guille_neosdata_npy_relative --base_filenames_file base_filenames.txt --learning_rate 5e-7 --continue_train --load_weights_only --no_load_hparams --load_optimizer_states --lr_policy none 
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_kulzaworld_neosraw_rel --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/kulzaworld_guille_neosdata_npy_relative --base_filenames_file base_filenames.txt --gradient_clip_val 10.0 --learning_rate 1e-5 --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_kulzaworld_neosraw_rel_nonshuff --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/kulzaworld_guille_neosdata_npy_relative --base_filenames_file base_filenames.txt --gradient_clip_val 20.0 --learning_rate 5e-6 --not_shuffle

	sbatch slurm_script4s.slurm $exp --hparams_file=training/hparams/neos_qb/${exp}.yaml --experiment_name ${exp}_quantum_bar_rel_nodp --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/quantum_bar_neosdata1_npy_relative --base_filenames_file base_filenames.txt
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_dekaworld_neosraw_rel --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/dekaworld_alex_guille_neosdata_npy_relative --base_filenames_file base_filenames.txt

	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_kulzaworld_neosraw_fixed --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/kulzaworld_guille_neosdata_npy2 --base_filenames_file base_filenames.txt --continue_train




	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_kulzaworld_neosraw --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/kulzaworld_guille_neosdata_npy2 --base_filenames_file base_filenames.txt --gradient_clip_val 10.0 --learning_rate 1e-5 --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_kulzaworld_neosraw_axis_angle --num_nodes 1 --max_epochs 6000 --data_dir=$SCRATCH/data/kulzaworld_guille_neosdata_npy_axis_angle --base_filenames_file base_filenames.txt --gradient_clip_val 10.0 --learning_rate 1e-5 --continue_train

	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_neosraw --num_nodes 1 --max_epochs 2000 --data_dir=$SCRATCH/data/dekaworld_alex_guille_neosdata --base_filenames_file base_filenames.txt --gradient_clip_val 10.0 --learning_rate 1e-5 --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_neosraw_lr2 --num_nodes 1 --max_epochs 2000 --data_dir=$SCRATCH/data/dekaworld_alex_guille_neosdata --base_filenames_file base_filenames.txt --gradient_clip_val 100.0 --learning_rate 3e-6 --continue_train

	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_gc10_lr1a_aistppkth_lr2 --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered8.txt --gradient_clip_val 10.0 --learning_rate 7e-5 --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_gc10_lr1a_aistpp --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered3.txt --gradient_clip_val 10.0 --learning_rate 1e-4 --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_gc10_lr1 --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --gradient_clip_val 10.0 --learning_rate 1e-5 --continue_train

	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_test --num_nodes 1 --max_epochs 1000 --data_dir=$SCRATCH/data/dance_combined2
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_test --num_nodes 1 --max_epochs 1000 --data_dir=$SCRATCH/data/dance_combined3
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_aistpp --num_nodes 1 --max_epochs 1000 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file crossmodal_train_filtered2.txt
	#sbatch slurm_script4.slurm $exp --experiment_name ${exp}_aistpp --num_nodes 1 --max_epochs 1000 --data_dir=$SCRATCH/data/dance_aistpp
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_aistpp --num_nodes 1 --max_epochs 1000 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file crossmodal_train_filtered2.txt --continue_train
	#sbatch slurm_script4.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1 --max_epochs 1000 --data_dir=$SCRATCH/data/dance_combined3
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --learning_rate 1e-4
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_wcos --num_nodes 1 --max_epochs 100 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --learning_rate 1e-3 --lr_policy LinearWarmupCosineAnnealing --warmup_epochs 1
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_exp_step --num_nodes 1 --max_epochs 100 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --learning_rate 1e-3 --lr_policy exponential_step --lr_decay_iters 1000 --lr_decay_factor 0.5
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_exp --num_nodes 1 --max_epochs 100 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --learning_rate 1e-3 --lr_policy exponential --lr_decay_factor 0.5 --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_lr2 --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --learning_rate 5e-4 --continue_train	
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_lr2_aistpp --num_nodes 1 --max_epochs 400 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered3.txt --learning_rate 5e-4 --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_aistpp --num_nodes 1 --max_epochs 400 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered3.txt --learning_rate 1e-4 --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_lr2_nojd --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered4.txt --learning_rate 5e-4 --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_lr2_nojd_nosyr --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered5.txt --learning_rate 5e-4
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_lr2_aistppkthmisc --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered8.txt --learning_rate 5e-4 --continue_train
	#sbatch slurm_script1s.slurm $exp --experiment_name ${exp}_newdata_filtered_lr2_aistppkthmisc_smol --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered8.txt --learning_rate 5e-4
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered --num_nodes 1 --max_epochs 2000 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_gc --num_nodes 1 --max_epochs 3000 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --gradient_clip_val 0.1 --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_gc10_short --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --gradient_clip_val 10.0
	#sbatch slurm_script4.slurm $exp --experiment_name ${exp}_newdata_filtered_gc10_short --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --gradient_clip_val 10.0 --continue_train
#	sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_gc10_lr1 --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --gradient_clip_val 10.0 --learning_rate 1e-5
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_gc10_lr1 --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --gradient_clip_val 10.0 --learning_rate 2e-5
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_gc10_lr1_aistpp --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered3.txt --gradient_clip_val 10.0 --learning_rate 1e-3 --lr_policy exponential_step --lr_decay_iters 2000 --lr_decay_factor 0.5
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_gc10_lr1b --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --gradient_clip_val 10.0 --learning_rate 1e-4 --lr_policy multistep --lr_decay_milestones '[1000,2000,4000,8000,16000]' --lr_decay_factor 0.5 --scheduler_interval step
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_gc10_lr1b_aistppkth --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered8.txt --gradient_clip_val 10.0 --learning_rate 1e-4 --lr_policy multistep --lr_decay_milestones '[8000,16000,32000,64000]' --lr_decay_factor 0.5 --scheduler_interval step
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_gc10_lr1a_aistppkth --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered8.txt --gradient_clip_val 10.0 --learning_rate 1e-4
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_gc10_lr1a_aistppkth_gc5 --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered8.txt --gradient_clip_val 5.0 --learning_rate 1e-4
	#sbatch slurm_script4.slurm $exp --experiment_name ${exp}_newdata_filtered_gc_short --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --gradient_clip_val 1.0
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_gc_short --num_nodes 1 --max_epochs 2000 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --gradient_clip_val 1.0 --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered --num_nodes 1 --max_epochs 100 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --gradient_clip_val 10 --continue_train --load_weights_only --no_load_hparams
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered --num_nodes 1 --max_epochs 100 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --gradient_clip_val 10 --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_gc_short2 --num_nodes 1 --max_epochs 100 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --gradient_clip_val 0.1 --continue_train --load_weights_only --no_load_hparams --batch_size 84 --learning_rate 5e-4
	#sbatch slurm_script4.slurm $exp --experiment_name ${exp}_newdata_filtered_gc_short --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --gradient_clip_val 0.1 --continue_train
#	sbatch slurm_script4.slurm $exp --experiment_name ${exp}_newdata_filtered_gc_short --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --gradient_clip_val 10 --continue_train --learning_rate 1e-6 --load_weights_only
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_gc_short --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --gradient_clip_val 10 --continue_train --learning_rate 1e-6
	#sbatch slurm_script4.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1 --max_epochs 1000 --data_dir=$SCRATCH/data/dance_combined3 --continue_train
	#sbatch slurm_script4.slurm $exp --experiment_name ${exp}_aistpp --num_nodes 1 --max_epochs 1000 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file crossmodal_train_filtered2.txt --continue_train
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_aistpp --num_nodes 1 --max_epochs 1000 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file crossmodal_train_filtered2.txt --continue_train
	#sbatch slurm_script.slurm $exp --experiment_name ${exp}_newdata4 --num_nodes 8 --continue_train --no_load_hparams 
	#sbatch slurm_script.slurm $exp --experiment_name ${exp}_newdata2 --num_nodes 8 --continue_train

	#sbatch slurm_script2.slurm $exp --experiment_name ${exp}_posemb --num_nodes 1 --data_dir=${SCRATCH}/data/dance_combined --continue_train
	#sbatch slurm_script2.slurm $exp --experiment_name ${exp}_newdata2 --num_nodes 1 --continue_train --hparams_file=training/hparams/dance_combined/moglow_expmap2.yaml
	#sbatch slurm_script2b.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1 --continue_train --data_dir=${SCRATCH}/data/dance_combined2

	#sbatch slurm_script2b.slurm $exp --experiment_name ${exp}_newdata2 --num_nodes 1
	#sbatch slurm_script2b.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1 --continue_train 
	#sbatch slurm_script2b.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1 --continue_train --base_filenames_file base_filenames_train_nojd.txt
	#sbatch slurm_script1.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1 #--fix_lengths
	#sbatch slurm_script1.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1 --continue_train
	#sbatch slurm_script1.slurm $exp --experiment_name ${exp}_newdata2 --num_nodes 1
	#sbatch slurm_script1.slurm $exp --experiment_name ${exp}_newdata2 --num_nodes 1 --max_epochs 10 --continue_train
	#sbatch slurm_script1.slurm $exp --experiment_name ${exp}_newdata2 --num_nodes 1 --continue_train --lr_decay_milestones="[1,3]"
	#sbatch slurm_script1.slurm $exp --experiment_name ${exp}_newdata2 --num_nodes 1 --continue_train --lr_decay_milestones="[2,3,5]"
	#sbatch slurm_script1.slurm $exp --experiment_name ${exp}_newdata2 --num_nodes 1 --continue_train --max_epochs 5
	#sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1 --max_epochs 300
	#sbatch slurm_script4.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1 --max_epochs 300
	#sbatch slurm_script4.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1 --max_epochs 300 --continue_train

	#sbatch slurm_script2.slurm $exp --experiment_name=${exp}_smoldata --data_dir=${SCRATCH}/data/dance_combined --base_filenames_file base_filenames_train_finetune.txt
done

