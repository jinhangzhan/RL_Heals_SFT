#!/bin/bash

#SBATCH --job-name=TrainVI-llama-50
#SBATCH --mail-user=xxx@email.com
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=500000M
#SBATCH --time=24:00:00
#SBATCH --output=./slurm_logs/train_%A_%a.out
#SBATCH --error=./slurm_logs/train_%A_%a.err
#SBATCH --gres=gpu:h100:4

module load cuda/12.2
module load cudnn/8.9.5.29
export CUDA_HOME=/path/Core/cudacore/12.2.2

export PYTHONPATH=/path/RL_Heals_SFT:$PYTHONPATH

#export OMP_NUM_THREADS=8

export WANDB_API_KEY="wandbKey"

CONDA_DIR="conda"
if [ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
    conda activate SFTvsRL3.13
    echo "Activated conda environment SFTvsRL3.13."
else
    echo "Conda activation script not found. Exiting."
    exit 1
fi

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4

LR=1e-6
save_every=10
save_model=True # disable running saving. one checkpoint ~30GB
check_point=50

CKPT_NAME="/path/checkpoint-${check_point}"
PORT=$((RANDOM % 10000 + 1000))

# download from our huggingface dataset repo tianzhechu/SFTvsRL_Data
ROUTE_INFO="/path/data/SFTvsRL_Data/VIRL_routes/nyc_1k_routes/route_infos.json" # .json
GPS_TO_PANO="/path/data/SFTvsRL_Data/VIRL_routes/nyc_1k_routes/gps_pano_mapping.pkl" # .pkl
STREETVIEWS="/path/data/SFTvsRL_Data/VIRL_routes/nyc_1k_routes/street_views/" # folder of images

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((10000 + RANDOM % 20000))
export MASTER_ADDR MASTER_PORT

echo "HOST=$HOSTNAME CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
#nvidia-smi --query-gpu=index,name --format=csv

echo "RANK=${SLURM_PROCID}"
DS_SKIP_CUDA_CHECK=1
TOKENIZERS_PARALLELISM=false

accelerate launch \
  --num_processes 4 \
  --num_machines $SLURM_JOB_NUM_NODES \
  --mixed_precision bf16 \
  --main_process_port=$MASTER_PORT \
  --main_process_ip=$MASTER_ADDR \
  --use_deepspeed \
  --deepspeed_config_file /path/RL_Heals_SFT/scripts/ds_zero2_offload.json \
  --deepspeed_multinode_launcher pdsh \
  --deepspeed_hostfile "$HOSTFILE" \
  -m rl.launcher \
  -f /path/RL_Heals_SFT/rl/configs/llama_virl_language.yaml \
  --report_to=none \
  --output_dir=/path/data/train_ckpt/virl_l/${check_point}/ \
  --optimizer_config.init_lr=${LR} \
  --optimizer_config.lr_max_steps=20 \
  --prompt_config.enable_verification=True \
  --num_updates=15 \
  --run_name=virl_language_training \
  --num_steps=256 \
  --model_path=${CKPT_NAME} \
  --save_ckpt=${save_model} \
  --env_config.route_info_path=${ROUTE_INFO} \
  --env_config.platform_cfg.OFFLINE.PANORAMA_DIR=${STREETVIEWS} \
  --env_config.platform_cfg.OFFLINE.GPS_TO_PANO_PATH=${GPS_TO_PANO} \
  --save_every=${save_every}  2>&1 | tee language-train-ck-${today}-${check_point}.log



