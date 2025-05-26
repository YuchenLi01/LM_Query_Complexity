#!/bin/bash
#SBATCH --job-name=codellama_err_pred_20250127_noback_trained
#SBATCH --output=codellama_err_pred_20250127_noback_trained.out
#SBATCH --error=codellama_err_pred_20250127_noback_trained.err
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
python src/main_run_codellama.py \
  model.name="codellama/CodeLlama-7b-hf" \
  generation_configs.backtrack_quota=0 \
  generation_configs.backtrack_stride=0 \
  generation_configs.top_p=1.0 \
  generation_configs.temperature=0.1 \
  generation_configs.block_err_pred=False \
  generation_configs.err_pred_threshold=0.9 \
  error_predictor.use_groundtruth=False \
  error_predictor.layer=27 \
  error_predictor.num_mlp_layer=1 \
  fs.prompts_and_generations_local_path="" \
  seed=42
