#!/bin/bash
#SBATCH --job-name=bandit_8gpu
#SBATCH --output=/mnt/project_modelware/zhaojian/weiyu/bandit_reinforce/output/slurm_%j.out
#SBATCH --error=/mnt/project_modelware/zhaojian/weiyu/bandit_reinforce/output/slurm_%j.err
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1

# Load environment
source activate roll 2>/dev/null || conda activate roll 2>/dev/null || true

# Run training
cd /mnt/project_modelware/zhaojian/weiyu/bandit_reinforce/bandit_reinforce-/experiments
bash run.sh exp1_bandit_ts_dapo_8gpu
