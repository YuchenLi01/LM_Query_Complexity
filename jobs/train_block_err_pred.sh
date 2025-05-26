#!/bin/bash
#SBATCH --job-name=train_4layer_block_err_pred_layer27
#SBATCH --output=train_4layer_block_err_pred_layer27.out
#SBATCH --error=train_4layer_block_err_pred_layer27.err
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=16G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:L40S:1

# Your job commands go here
cd ~/LM_Query_Complexity/
conda init
conda activate handbook
module load cuda-12.1
export NCCL_P2P_DISABLE=1
python src/train_err_pred.py \
  --layer 27 \
  --block True \
  --errors_hiddens_path "data/block_errors_hiddens_layer27.pkl" \
  --correct_hiddens_path "data/block_correct_hiddens_layer27.pkl" \
  --best_model_err_pred_save_path "data/model_4layer_block_err_pred_layer27.pkl" \
  --err_pred_num_layers 4
