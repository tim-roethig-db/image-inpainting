#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=MLprak

#resources:

#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=day
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=20G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:2
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=20:00:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"


# After J, specify name of the job to avoid confusion
#SBATCH --error=job.%J_sampleSize.err
# write the error output to job.*jobID*.err

#SBATCH --output=job.%J_sampleSize.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=m.blum@student.uni-tuebingen.de
# your mail address

####
#b) copy all needed data to the jobs scratch folder
####
cp -R /home/mollik/Image-Inpainting/data /scratch/$SLURM_JOB_ID/

####
#c) Execute your file.
####

# Uncomment to enable sbatch file
singularity exec --nv /common/singularityImages/default.simg /home/mollik/miniconda3/envs/inpainting/bin/python train.py

echo DONE!
