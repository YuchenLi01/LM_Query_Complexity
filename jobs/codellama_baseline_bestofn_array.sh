#!/bin/bash
#SBATCH --job-name=codellama_bon_baseline
#SBATCH --output=codellama_bon_baseline_%A_bon%a.out
#SBATCH --error=codellama_bon_baseline_%A_bon%a.err
#SBATCH --array=2,4,8
#SBATCH --partition=array
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

for err_pred_threshold in 0.9
do
  for top_p in 0.95
  do
    for exp_idx in 0 1 2 3 4
    do
      python src/main_run_codellama.py \
        model.name="codellama/CodeLlama-7b-hf" \
        generation_configs.backtrack_quota=0 \
        generation_configs.backtrack_stride=0 \
        generation_configs.top_p=$top_p \
        generation_configs.temperature=1.0 \
        generation_configs.block_err_pred=True \
        generation_configs.err_pred_threshold="$err_pred_threshold" \
        generation_configs.block_best_of_n="${SLURM_ARRAY_TASK_ID}" \
        error_predictor.use_groundtruth=False \
        error_predictor.layer=27 \
        error_predictor.num_mlp_layer=1 \
        fs.prompts_and_generations_local_path="" \
        seed=$exp_idx
    done
  done
done
