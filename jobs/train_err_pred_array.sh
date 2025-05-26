#!/bin/bash
#SBATCH --job-name=train_err_pred_layer
#SBATCH --output=train_err_pred_layer_%A_%a.out
#SBATCH --error=train_err_pred_layer_%A_%a.err
#SBATCH --array=24,25,26
#SBATCH --partition=array
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=16G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:L40S:1

# Your job commands go here
cd ~/lm_ft_data/
conda init
conda activate handbook
module load cuda-12.1
export NCCL_P2P_DISABLE=1
python src/train_err_pred.py \
  --layer "${SLURM_ARRAY_TASK_ID}" \
  --err_pred_num_layers 1
