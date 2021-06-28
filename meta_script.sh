#!/bin/bash

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
for exp in moglow_expmap1_label4d
#for exp in moglow_expmap1_label4e
#for exp in transflower_expmap_cr4_label_bs5c
#for exp in transflower_expmap_cr4_label_bs5 transflower_expmap_cr
#for exp in transflower_expmap_cr4_label_bs5
#for exp in transflower_expmap_cr4_label_bs5d
#for exp in transflower_expmap_cr_label2
#for exp in transflower_expmap_cr_label3
#for exp in transflower_expmap_cr_label5
do
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
	sbatch slurm_script1.slurm $exp --experiment_name ${exp}_newdata2 --num_nodes 1 --max_epochs 10 --continue_train
	#sbatch slurm_script1.slurm $exp --experiment_name ${exp}_newdata2 --num_nodes 1 --continue_train --lr_decay_milestones="[1,3]"
	#sbatch slurm_script1.slurm $exp --experiment_name ${exp}_newdata2 --num_nodes 1 --continue_train --lr_decay_milestones="[2,3,5]"
	#sbatch slurm_script1.slurm $exp --experiment_name ${exp}_newdata2 --num_nodes 1 --continue_train --max_epochs 5
	#sbatch slurm_script4.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1 --max_epochs 300
	#sbatch slurm_script4.slurm $exp --experiment_name ${exp}_newdata --num_nodes 1 --max_epochs 300 --continue_train

	#sbatch slurm_script2.slurm $exp --experiment_name=${exp}_smoldata --data_dir=${SCRATCH}/data/dance_combined --base_filenames_file base_filenames_train_finetune.txt
done

