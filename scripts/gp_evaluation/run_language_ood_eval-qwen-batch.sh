#!/bin/bash
set -euo pipefail

module load cuda/12.2.2/cudnn/8.9
export CUDA_HOME=/path/cudacore/12.2.2
export WANDB_API_KEY="wandbKey"
export PYTHONPATH=/path/RL_Heals_SFT:$PYTHONPATH

CONDA_DIR="/path/conda"
source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate SFTvsRL3.13

export WANDB_MODE=offline

VITER=5
ENABLE=True
OOD=True
FACE10=False
TARGET=24
NUM_TRAJ=234
TODAY=$(date '+%Y-%m-%d')

eval_ckpt () {
    GPU_ID=$1
    CKPT=$2
    export CUDA_VISIBLE_DEVICES=${GPU_ID}

    CKPT_NAME="/path/gp_l_sft-qwen/checkpoint-${CKPT}"
    OUTPUT_DIR="logs/gp_l_ood_verify_${VITER}_target_${TARGET}_${TODAY}_${CKPT}"
    PORT=$((RANDOM % 10000 + 1000 + GPU_ID))

    accelerate launch \
      --config_file /path/RL_Heals_SFT/scripts/config_zero2_1gpu.yaml \
      --main_process_port ${PORT} \
      -m evaluation.launcher \
      -f /path/RL_Heals_SFT/evaluation/configs/llama_gp_language-qwen.yaml \
      --model_path=${CKPT_NAME} \
      --output_dir=${OUTPUT_DIR}/gp_l_ood-qwen.jsonl \
      --prompt_config.enable_verification=${ENABLE} \
      --env_config.target_points=${TARGET} \
      --env_config.verify_iter=${VITER} \
      --env_config.treat_face_cards_as_10=${FACE10} \
      --env_config.ood=${OOD} \
      --num_traj=${NUM_TRAJ} \
      2>&1 | tee "eval-ood-qwen-${TODAY}-ck${CKPT}.log"
}
export -f eval_ckpt

GPUS=(0 1 2 3)
CKPTS=(100 200 300 400)

parallel --jobs ${#GPUS[@]} \
         eval_ckpt ::: "${GPUS[@]}" ::: "${CKPTS[@]}"