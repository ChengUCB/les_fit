#!/bin/bash
#
#----------------------------------
# single GPU + single CPU example
#----------------------------------
#
#SBATCH --job-name=test
#
#Define the number of hours the job should run.
#Maximum runtime is limited to 10 days, ie. 240 hours
#SBATCH --time=48:00:00
#
#Define the amount of system RAM used by your job in GigaBytes
#SBATCH --mem=45G
#
#Pick whether you prefer requeue or not. If you use the --requeue
#option, the requeued job script will start from the beginning,
#potentially overwriting your previous progress, so be careful.
#For some people the --requeue option might be desired if their
#application will continue from the last state.
#Do not requeue the job in the case it fails.
#SBATCH --no-requeue
#
#Define the "gpu" partition for GPU-accelerated jobs
#SBATCH --partition=gpu100
##SBATCH --exclude=gpu[113,114,118,119,123-127,136-139,144-148,150]
#
#Define the number of GPUs used by your job
#SBATCH --gres=gpu:1
#
#Define the GPU architecture (GTX980 in the example, other options are GTX1080Ti, K40)
##SBATCH --constraint=GTX980
#
#Do not export the local environment to the compute nodes
##SBATCH --export=NONE
##unset SLURM_EXPORT_ENV
#

source ~/.bashrc
#load an CUDA software module
module load cuda/11.8.0

python fit-cace-nnp.py

wait
