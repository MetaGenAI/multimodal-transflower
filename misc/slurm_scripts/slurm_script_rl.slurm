#!/bin/bash

#SBATCH --job-name=ppo_test
##SBATCH --time=100:00:00
##SBATCH --time=20:00:00
#SBATCH --time=01:00:00
##SBATCH --qos=qos_gpu-t4
##SBATCH --qos=qos_gpu-t3
#SBATCH --qos=qos_gpu-dev
##SBATCH --ntasks=1 --cpus-per-task=24 --gres=gpu:4
##SBATCH --ntasks=1 --cpus-per-task=24 --gres=gpu:1
##SBATCH --nodes=4 --ntasks-per-node=4 --cpus-per-task=6 --gres=gpu:4
##SBATCH --nodes=2 --ntasks-per-node=4 --cpus-per-task=6 --gres=gpu:4
#SBATCH --nodes=1 --ntasks-per-node=20 --cpus-per-task=2 --gres=gpu:4
##SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=6 --gres=gpu:2
##SBATCH --partition=gpu_p2
#SBATCH -A imi@v100
##SBATCH -A imi@gpu
#SBATCH -C v100-32g
#SBATCH --exclusive
#SBATCH --signal=TERM@120

export MASTER_PORT=1234
slurm_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
echo $slurm_nodes
export MASTER_ADDRESS=$(echo $slurm_nodes | cut -d' ' -f1)
echo $MASTER_ADDRESS
export ROOT_DIR_MODEL=/gpfswork/rech/imi/usc19dv/mt-lightning/

export PYTHONPATH=${PYTHONPATH}:/gpfswork/rech/imi/usc19dv/lib/python3.7/site-packages:/gpfsscratch/rech/imi/usc19dv/lib/python3.7/site-packages/
export ROOT_FOLDER=/gpfswork/rech/imi/usc19dv/captionRLenv/
export DATA_FOLDER=/gpfsscratch/rech/imi/usc19dv/data/UR5/
export PROCESSED_DATA_FOLDER=/gpfsscratch/rech/imi/usc19dv/data/UR5_processed/
export ROOT_DIR_MODEL=/gpfswork/rech/imi/usc19dv/mt-lightning/
export PRETRAINED_FOLDER=/gpfswork/rech/imi/usc19dv/mt-lightning/training/experiments/
export ROOT_GENERATED_DATA=/gpfsscratch/rech/imi/usc19dv/data/


module purge
#module load pytorch-gpu/py3/1.8.0
module load pytorch-gpu/py3/1.10.1

#exp=$1
#srun ./script_train.sh $@
srun -n 1 python ./inference/sandbox_rl.py --n 20
#srun ./script_train_dev.sh $exp
