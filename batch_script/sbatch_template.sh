#!/bin/bash
#SBATCH --job-name=pCR_MLP_Grid
#SBATCH -o /home/aaron.l/Pillar/pillar-pretrain/Breast-MRI-pCR/log/%x_%j.stdout.txt
#SBATCH -e /home/aaron.l/Pillar/pillar-pretrain/Breast-MRI-pCR/log/%x_%j.stderr.txt
#SBATCH --partition=tier2_gpu
#SBATCH --qos=normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --account=aimilia_gastounioti
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aaron.l@wustl.edu

# Clean environment
unset VIRTUAL_ENV
unset PYTHONHOME
unset PYTHONPATH

# Move to project directory
cd /home/aaron.l/Pillar/pillar-pretrain/Breast-MRI-pCR

# WandB Setup
export WANDB_MODE=online
export WANDB_API_KEY='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'

# Run with passed arguments
# The $STACK variable will contain "--stack_rgb" or remain empty
uv run python run.py \
    --lr $LR \
    --batch_size $BS \
    --residual_dropout $RD \
    --head_dropout $HD \
    $STACK