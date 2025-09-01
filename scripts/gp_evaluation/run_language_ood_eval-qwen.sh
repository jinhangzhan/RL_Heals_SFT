#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=3

module load cuda/12.2.2/cudnn/8.9
export CUDA_HOME=/path/cudacore/12.2.2
export WANDB_API_KEY="wandbKey"
export PYTHONPATH=/path/RL_Heals_SFT:$PYTHONPATH

CONDA_DIR="/path/conda"
if [ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
    conda activate SFTvsRL3.13
    echo "Activated conda environment SFTvsRL3.13."
else
    echo "Conda activation script not found. Exiting."
    exit 1
fi

VITER=5
ENABLE=True
OOD=True
FACE10=False
TARGET=24
CKPT=120
today=$(date '+%Y-%m-%d')
NUM_TRAJ=234

k_top=64
k_tail=0
CKPT_NAME="/path/checkpoint-${CKPT}"
OUTPUT_FOLDER="logs/gp_l_ood_verify_${VITER}_target_${TARGET}_${today}_${CKPT}"
PORT=$((RANDOM % 10000 + 1000))

export NEW_AVX_FLAGS="-mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mavx512ifma -mavx512vbmi -D__AVX512__"
export CXXFLAGS_ORIG="${CXXFLAGS}"
if [ -n "${CXXFLAGS_ORIG}" ]; then
  export CXXFLAGS="${CXXFLAGS_ORIG} ${NEW_AVX_FLAGS}"
else
  export CXXFLAGS="${NEW_AVX_FLAGS}"
fi

export CXXFLAGS_ORIG="${CXXFLAGS}"
if [ -n "${CXXFLAGS_ORIG}" ]; then
  export CXXFLAGS="${CXXFLAGS_ORIG} ${AVX_CPU_FLAGS}"
else
  export CXXFLAGS="${AVX_CPU_FLAGS}"
fi

echo "Running with CFLAGS=${CFLAGS}"
echo "Running with CXXFLAGS=${CXXFLAGS}"

accelerate launch  \
    --config_file /path/RL_Heals_SFT/scripts/config_zero2_1gpu.yaml \
    --main_process_port ${PORT} \
    -m evaluation.launcher \
    -f /path/RL_Heals_SFT/evaluation/configs/llama_gp_language-qwen.yaml \
    --model_path=${CKPT_NAME} \
    --output_dir=${OUTPUT_FOLDER}/gp_l_ood.jsonl \
    --prompt_config.enable_verification=${ENABLE} \
    --env_config.target_points=${TARGET} \
    --env_config.verify_iter=${VITER} \
    --env_config.treat_face_cards_as_10=${FACE10} \
    --env_config.ood=${OOD} \
    --num_traj=${NUM_TRAJ} \
 2>&1 | tee "eval-ood-qwen-${TODAY}-ck${CKPT}.log"
# 2>&1 | tee "eval-ood-rl-qwen-800-4c-${TODAY}-ck${CKPT}.log"

