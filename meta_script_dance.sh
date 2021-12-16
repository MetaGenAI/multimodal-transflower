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
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss4 transflower_expmap_cr4_bs5_og2_futureN_gauss5
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss5
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss6
#for exp in moglow_expmap1_tf2
#for exp in moglow_expmap1_tf3
#for exp in mowgli_expmapb
#for exp in mowgli_expmap_nocond
#for exp in mowgli_expmap_nocond2
#for exp in mowgli_expmap_nocond4
#for exp in mowgli_expmap_nocond5
#for exp in mowgli_expmap_nocond_output_chunking
#for exp in mowgli_expmap_nocond_output_chunking2
#for exp in mowgli_expmap_nocond_output_chunking2_stage2
#for exp in mowgli_expmap_nocond_output_chunking2b_stage2
#for exp in mowgli_expmap_nocond_output_chunking3
#for exp in mowgli_expmap_nocond_output_chunking3_stage2
#for exp in mowgli_expmap_nocond_output_chunking3_stage2 mowgli_expmap_nocond_output_chunking2_stage2 
#for exp in mowgli_expmap_nocond_output_chunking4
#for exp in mowgli_expmap_nocond_output_chunking5
for exp in mowgli_expmap_nocond_output_chunking6
#for exp in mowgli_expmap_nocond_output_chunking6_stage2
#for exp in mowgli_expmap_nocond_output_chunking3b_stage2
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss_simon
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss_bn
#for exp in transflower_expmap_cr4_bs5_og2_futureN_gauss_60
#for exp in transformer_expmap_cr_60
#for exp in transformer_expmap_cr
#for exp in transformer_expmap_cr transformer_expmap_cr_N7
#for exp in transformer_expmap_cr_N7
#for exp in transformer_expmap_cr_N7
#for exp in transformer_expmap_simon
#for exp in transflower_expmap_cr4_label_bs5d
#for exp in transflower_expmap_cr_label2
#for exp in transflower_expmap_cr_label3
#for exp in transflower_expmap_cr_label5

do

	#sbatch slurm_script4s.slurm $exp --hparams_file=training/hparams/dance_combined/${exp}.yaml --experiment_name ${exp}_newdata_filtered_aistpp --num_nodes 1 --max_epochs 3000 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered3.txt --continue_train --load_weights_only --no_load_hparams
	sbatch slurm_script4s.slurm $exp --hparams_file=training/hparams/dance_combined/${exp}.yaml --experiment_name ${exp}_newdata_filtered_aistpp --num_nodes 1 --max_epochs 3000 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered3.txt --continue_train --load_weights_only --no_load_hparams

#	sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_aistpp --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered3.txt --learning_rate 1e-4 --continue_train
#	sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_aistppkth --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered8.txt --learning_rate 1e-4 --continue_train
#	sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered2.txt --learning_rate 1e-4 --continue_train
#	sbatch slurm_script4s.slurm $exp --experiment_name ${exp}_newdata_filtered_lr2_sm --num_nodes 1 --max_epochs 300 --data_dir=$SCRATCH/data/dance_combined3 --base_filenames_file base_filenames_train_filtered9.txt --learning_rate 5e-4 --continue_train


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

