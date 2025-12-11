#!/bin/bash
#SBATCH --job-name=MACELES_training
#SBATCH --account=co_12monkeys
#SBATCH --partition=savio4_gpu
##SBATCH --mem=48G
#SBATCH --qos=12monkeys_gpu4_normal
#SBATCH --cpus-per-task=8 --gres=gpu:L40:1 --ntasks=1
#SBATCH --time=20:00:00
#SBATCH -o out.%j
#SBATCH -e err.%j

echo "Start: $(date)"
echo "cwd: $(pwd)"
source ~/.bashrc

conda activate maceles

module load gcc/11.4.0 openmpi/4.1.6
module load cuda/11.8.0

TRAIN_SCRIPT="/global/scratch/users/dongjinkim/md_jobs/MACELES-OFF/mace/scripts/run_train.py"

DIRS=(
  "mace-les-r45-nl2-l1_5"
  "mace-les-r45-nl2-l1_10"
  "mace-les-r45-nl2-l1_20"
  "mace-les-r45-nl2-l1_30"
  "mace-les-r45-nl2-l1_50"
  "mace-les-r45-nl2-l1_100"
  "mace-les-r45-nl2-l1_200"
  "mace-les-r45-nl2-l1_300"
  "mace-les-r45-nl2-l1_400"
  "mace-les-r45-nl2-l1_500"
)

for d in "${DIRS[@]}"; do
    if [ ! -d "$d" ]; then
        echo "Skip: $d (directory not found)"
        continue
    fi

    echo "========================================"
    echo "Training in directory: $d"
    echo "========================================"

    cd "$d"

    python "$TRAIN_SCRIPT" \
        --name="H2O" \
        --train_file="./train.xyz" \
        --valid_fraction=0.05 \
        --test_file="../../test-H2O_RPBE-D3.xyz" \
        --energy_key="energy" \
        --forces_key="forces" \
        --E0s='average' \
        --model="MACELES" \
        --hidden_irreps='128x0e+ 128x1o' \
        --r_max=4.5 \
        --num_interactions=2 \
        --batch_size=4 \
        --max_num_epochs=1000 \
        --stage_two \
        --start_stage_two=500 \
        --ema \
        --ema_decay=0.99 \
        --amsgrad \
        --restart_latest \
        --device=cuda \
        --default_dtype="float32"

    python /global/scratch/users/dongjinkim/md_jobs/MACELES-OFF/mace/scripts/eval_configs.py --configs ../h2o_bec.xyz --output test-mace.xyz --model H2O_stagetwo.model --batch_size 1 --default_dtype float32 --compute_bec

    cd ..

    echo "Finished: $d"
done

wait
echo "End: $(date)"
