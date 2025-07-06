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
#SBATCH --time=24:00:00
#
#Define the amount of system RAM used by your job in GigaBytes
#SBATCH --mem=48G
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

python /nfs/scistore23/chenggrp/bcheng/cace-leslib/mace/scripts/run_train.py \
    --name="Au2-MgO" \
    --train_file="../train-Au-MgO-Al.xyz" \
    --valid_fraction=0.05 \
    --test_file="../test-Au-MgO-Al.xyz" \
    --energy_key="energy" \
    --forces_key="forces" \
    --E0s='average' \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=4.5 \
    --num_interactions=2 \
    --batch_size=10 \
    --max_num_epochs=1000 \
    --stage_two \
    --start_stage_two=500 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --restart_latest \
    --device=cuda \
