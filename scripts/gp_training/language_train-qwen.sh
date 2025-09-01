#!/bin/bash

#SBATCH --job-name=TrainGP-qwen-800
#SBATCH --mail-user=xxx@email.com
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8                        # Total number of cores requested
#SBATCH --constraint=80gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=500G
#SBATCH --time=48:00:00                     # Time limit (hh:mm:ss)
#SBATCH --output=./slurm_logs/train_%A_%a.out
#SBATCH --error=./slurm_logs/train_%A_%a.err
#SBATCH --gres=gpu:a100l:4
#SBATCH --wait-all-nodes=1

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

export OMP_NUM_THREADS=8
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

unset SETUPTOOLS_USE_DISTUTILS
LR=1e-6
save_every=50
save_model=True # disable running saving. one checkpoint ~30GB

check_point=800
today=$(date -d "now" +%Y-%m-%d)

#CKPT_NAME="/network/scratch/l/luansito/data/GP-L-Init" # official init model: tianzhechu/GP-L-Init
#CKPT_NAME="/network/scratch/l/luansito/data/train_ckpt/gp_l_sft/checkpoint-${check_point}"
#CKPT_NAME="/network/scratch/l/luansito/data/train_ckpt/gp_l_sft/recover/checkpoint-1100"
CKPT_NAME="/network/scratch/l/luansito/data/train_ckpt/gp_l_sft_qwen/checkpoint-800"
#CKPT_NAME="/network/scratch/l/luansito/data/train_ckpt/gp_language/output_2025-04-17||23:29:40/checkpoint-epoch-${check_point}"
#CKPT_NAME="/network/scratch/l/luansito/data/train_ckpt/gp_language/output_2025-04-18||23:35:44/checkpoint-epoch-${check_point}/"
#PORT=$((RANDOM % 10000 + 1000))

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((10000 + RANDOM % 20000))
export MASTER_ADDR MASTER_PORT

echo "HOST=$HOSTNAME CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name --format=csv

echo "RANK=${SLURM_PROCID}"

accelerate launch \
  --num_processes 4 \
  --num_machines $SLURM_JOB_NUM_NODES \
  --mixed_precision bf16 \
  --main_process_port=$MASTER_PORT \
  --main_process_ip=$MASTER_ADDR \
  --use_deepspeed \
  --deepspeed_config_file /home/mila/l/luansito/kim/SFTvsRL/scripts/ds_zero2_offload.json \
  --deepspeed_multinode_launcher pdsh \
  --deepspeed_hostfile "$HOSTFILE" \
  -m rl.launcher \
  -f /home/mila/l/luansito/kim/SFTvsRL/rl/configs/llama_gp_language-qwen.yaml \
  --report_to=none \
  --output_dir="/network/scratch/l/luansito/data/train_ckpt/gp_language-qwen/${check_point}/" \
  --optimizer_config.init_lr=${LR} \
  --optimizer_config.lr_max_steps=300 \
  --prompt_config.enable_verification=True \
  --num_updates=300 \
  --env_config.treat_face_cards_as_10=True \
  --env_config.target_points=24 \
  --run_name=gp_language_training \
  --num_steps=32 \
  --model_path=${CKPT_NAME} \
  --save_ckpt=${save_model} \
  --seed=0 \
  --save_every=${save_every}  2>&1 | tee "language-train-ck-${today}-${check_point}.log"

