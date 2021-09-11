#!/bin/bash
#SBATCH --job-name=feature_extraction      # name of job
#SBATCH --ntasks=8                # total number of MPI processes
##SBATCH --ntasks-per-node=16       # number of MPI processes per node
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread       # 1 MPI process per physical core (no hyperthreading)
#SBATCH --time=00:10:00            # maximum execution time requested (HH:MM:SS)
##SBATCH --output=TravailMPI%j.out  # name of output file
##SBATCH --error=TravailMPI%j.out   # name of error file (here, in common with output)
#SBATCH --partition=cpu_p1
#SBATCH --account=imi@cpu
#SBATCH --mem-per-cpu=3gb
 
# go into the submission directory
#cd ${SLURM_SUBMIT_DIR}
 
# clean out the modules loaded in interactive and inherited by default
#module purge
 
# loading modules
#module load intel-all/19.0.4
module load openmpi/4.0.5-cuda
module load pytorch-gpu/py3/1.8.0
 
# echo of launched commands
set -x
 
# code execution
srun -n 16 ./feature_extraction/audio_feature_extraction_test.sh /gpfsscratch/rech/imi/usc19dv/data/aistpp_long_audios/ --replace_existing
