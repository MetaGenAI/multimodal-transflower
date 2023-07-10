#!/bin/bash
#SBATCH --job-name=feature_extraction      # name of job
#SBATCH --ntasks=6                # total number of MPI processes
##SBATCH --ntasks-per-node=16       # number of MPI processes per node
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
##SBATCH --hint=nomultithread       # 1 MPI process per physical core (no hyperthreading)
#SBATCH --time=02:00:00            # maximum execution time requested (HH:MM:SS)
##SBATCH --output=TravailMPI%j.out  # name of output file
##SBATCH --error=TravailMPI%j.out   # name of error file (here, in common with output)
#SBATCH --partition=cpu_p1
#SBATCH --account=imi@cpu
#SBATCH --mem-per-cpu=1gb
 
# go into the submission directory
#cd ${SLURM_SUBMIT_DIR}
export MASTER_PORT=1234
slurm_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
echo $slurm_nodes
export MASTER_ADDRESS=$(echo $slurm_nodes | cut -d' ' -f1)
echo $MASTER_ADDRESS

 
# clean out the modules loaded in interactive and inherited by default
module purge
 
# loading modules
#module load intel-all/19.0.4
#module load pytorch-gpu/py3/1.8.0
module load pytorch-cpu/py3/1.7.1
#module load intel-all/19.0.4
module load openmpi/4.0.5
 
# echo of launched commands
set -x

export n=6
 
# code execution
#srun -n 16 ./feature_extraction/audio_feature_extraction_test.sh /gpfsscratch/rech/imi/usc19dv/data/aistpp_long_audios/ --replace_existing
#srun -n 40 -pty bash -c "./feature_extraction/motion_feature_extraction.sh /gpfsscratch/rech/imi/usc19dv/data/dance_combined3 --replace_existing"
#./feature_extraction/motion_feature_extraction.sh /gpfsscratch/rech/imi/usc19dv/data/dance_combined3 --replace_existing
#./feature_extraction/motion_feature_extraction.sh /gpfsscratch/rech/imi/usc19dv/data/testing_tmp --replace_existing
srun -n $n -pty bash -c './feature_extraction/neos_feature_extraction.sh data/dekaworld_alex_guille_neosdata2 --replace_existing'
#./feature_extraction/motion_feature_extraction.sh /gpfsscratch/rech/imi/usc19dv/data/dance_combined3 --replace_existing
