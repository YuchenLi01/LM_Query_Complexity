#!/bin/bash
#SBATCH --job-name=codellama_err_pred_p0.95_ood
#SBATCH --output=codellama_err_pred_p0.95_ood_%A_mlp%a.out
#SBATCH --error=codellama_err_pred_p0.95_ood_%A_mlp%a.err
#SBATCH --array=0,1,2,3,4
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

for err_pred_threshold in 0.5 0.9
do
    python src/main_run_codellama.py \
      model.name="codellama/CodeLlama-7b-hf" \
      generation_configs.backtrack_quota=4 \
      generation_configs.backtrack_stride=4 \
      generation_configs.top_p=0.95 \
      generation_configs.temperature=1.0 \
      generation_configs.block_err_pred=False \
      generation_configs.err_pred_threshold="$err_pred_threshold" \
      error_predictor.use_groundtruth=False \
      error_predictor.layer=27 \
      error_predictor.num_mlp_layer=1 \
      fs.prompts_and_generations_local_path="['pop','add','sub','mul','div','max','min','std','avg','exp']" \
      seed="${SLURM_ARRAY_TASK_ID}"
done
