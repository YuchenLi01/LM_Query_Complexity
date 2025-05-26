#!/bin/bash
#SBATCH --job-name=train_dyck_lm_k2_0.2_0.8_m12_len32
#SBATCH --output=train_dyck_lm_k2_0.2_0.8_m12_len32.out
#SBATCH --error=train_dyck_lm_k2_0.2_0.8_m12_len32.err
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
source /usr/share/Modules/init/bash
module load cuda-12.1
export NCCL_P2P_DISABLE=1
python src/run_lm.py \
  --config "config/dyck_k2_0.2_0.8_m12_len32/dim512_depth12_heads8/batch32_lr0.0003_wd0.1_warmup100/config.yaml" \
  --train \
  --no-eval \
  --no-plot_representations
