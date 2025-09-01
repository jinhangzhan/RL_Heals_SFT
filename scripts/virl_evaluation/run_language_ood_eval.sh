#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
check_point=29
module load cuda/12.2
module load cudnn/8.9.5.29
export CUDA_HOME=/path/Core/cudacore/12.2.2

export PYTHONPATH=/path/RL_Heals_SFT:$PYTHONPATH

# W&B
export WANDB_API_KEY="wandbKey"

export WANDB_MODE=offline

CONDA_DIR="conda"
if [ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
    conda activate SFTvsRL3.13
    echo "Activated conda environment SFTvsRL3.13."
else
    echo "Conda activation script not found. Exiting."
    exit 1
fi

VITER=2

# enable verification
ENABLE=True
# use absolute action space, consistent with training
ABS=False
NUM_TRAJ=48
today=$(date +"%Y-%m-%d")
CKPT_NAME="/path/data/train_ckpt/virl_l/llama/50/output_2025-07-25||00:36:49/checkpoint-epoch-${check_point}"
OUTPUT_FOLDER="logs/virl_vl_rule_ood_verify_${VITER}"
PORT=$((RANDOM % 10000 + 2000))

# download from our huggingface dataset repo tianzhechu/SFTvsRL_Data
#ROUTE_INFO="YOUR_ROUTE_INFO_PATH" # .json
#GPS_TO_PANO="YOUR_GPS_TO_PANO_MAPPING_PATH" # .pkl
#STREETVIEWS="YOUR_STREETVIEWS_PATH" # folder of images
ROUTE_INFO="/path/data/SFTvsRL_Data/VIRL_routes/nyc_1k_routes/route_infos.json" # .json
GPS_TO_PANO="/path/data/SFTvsRL_Data/VIRL_routes/nyc_1k_routes/gps_pano_mapping.pkl" # .pkl
STREETVIEWS="/path/data/SFTvsRL_Data/VIRL_routes/nyc_1k_routes/street_views/" # folder of images

DS_SKIP_CUDA_CHECK=1 accelerate launch \
    --config_file /path/RL_Heals_SFT/scripts/config_zero2_1gpu.yaml --main_process_port ${PORT} \
    -m evaluation.launcher -f /path/RL_Heals_SFT/evaluation/configs/llama_virl_language.yaml \
    --model_path=${CKPT_NAME} \
    --output_dir=${OUTPUT_FOLDER}/virl_l_ood.jsonl \
    --env_config.route_info_path=${ROUTE_INFO} \
    --env_config.platform_cfg.OFFLINE.PANORAMA_DIR=${STREETVIEWS} \
    --env_config.platform_cfg.OFFLINE.GPS_TO_PANO_PATH=${GPS_TO_PANO} \
    --prompt_config.enable_verification=${ENABLE} \
    --env_config.verify_iter=${VITER} \
    --env_config.absolute_action=${ABS} \
    --num_traj=${NUM_TRAJ} \
  | tee eval-ood-llama-rl-50-${today}-${check_point}.log

