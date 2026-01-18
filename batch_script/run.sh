#!/bin/bash
#SBATCH --job-name=pCR_MLP_Folds
#SBATCH -o /home/aaron.l/Pillar/pillar-pretrain/Breast-MRI-pCR/log/%x_%j.stdout.txt
#SBATCH -e /home/aaron.l/Pillar/pillar-pretrain/Breast-MRI-pCR/log/%x_%j.stderr.txt
#SBATCH --partition=tier2_gpu
#SBATCH --qos=normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --account=aimilia_gastounioti
#SBATCH --mem=32G           # Increased for MRI processing
#SBATCH --cpus-per-task=4   # Match to num_workers in your DataLoader
#SBATCH --time=24:00:00      # 5 folds take time
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=aaron.l@wustl.edu

# 1. Clean up environment variables to stop the "VIRTUAL_ENV mismatch" warning
unset VIRTUAL_ENV
unset PYTHONHOME
unset PYTHONPATH

# 2. Use an ABSOLUTE path to move into your project folder
cd /home/aaron.l/Pillar/pillar-pretrain/Breast-MRI-pCR
export WANDB_MODE=online
export WANDB_API_KEY='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'
# 3. Use 'uv run' directly (it handles syncing and environment activation automatically)
# We don't need 'source .venv/bin/activate' if we use 'uv run'
uv run python run.py