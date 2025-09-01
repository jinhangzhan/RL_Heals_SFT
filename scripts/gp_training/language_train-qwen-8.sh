#!/bin/bash
#SBATCH --job-name=qwen-1000
#SBATCH --nodes=2
#SBATCH --gres=gpu:h100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=500000M
#SBATCH --output=slurm_logs/train_%j.out
#SBATCH --error=slurm_logs/train_%j.err

module load cuda/12.2
module load cudnn/8.9.5.29
source ~/miniconda3/etc/profile.d/conda.sh
conda activate SFTvsRL3.13
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH=/path/RL_Heals_SFT:$PYTHONPATH
export WANDB_MODE=disabled
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((10000 + RANDOM % 20000))

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

LR=1e-6
SAVE_EVERY=10
CKPT=/path
OUTDIR=/path/train_ckpt/gp_language-qwen

HOSTFILE=/path/RL_Heals_SFT/scripts/gp_training/hostfile_${SLURM_JOB_ID}
scontrol show hostnames "$SLURM_NODELIST" | awk '{print $1" slots=4"}' > "$HOSTFILE"
export ACCELERATE_HOST_FILE=$HOSTFILE
HOSTFILE_OPT="--hostfile $HOSTFILE"

hosts=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
echo hosts
for h in "${hosts[@]}"; do
  ssh-copy-id -i ~/.ssh/id_ed25519.pub $h
done

accelerate launch \
  --num_processes 8 \
  --num_machines $SLURM_JOB_NUM_NODES \
  --mixed_precision bf16 \
  --main_process_port=$MASTER_PORT \
  --main_process_ip=$MASTER_ADDR \
  --use_deepspeed \
  --deepspeed_config_file /path/RL_Heals_SFT/scripts/ds_zero2_offload.json \
  --deepspeed_multinode_launcher pdsh \
  --deepspeed_hostfile "$HOSTFILE" \
  -m rl.launcher \
    -f /path/RL_Heals_SFT/rl/configs/llama_gp_language-qwen.yaml \
    --output_dir=$OUTDIR \
    --report_to=none \
    --optimizer_config.init_lr=$LR \
    --optimizer_config.lr_max_steps=200 \
    --prompt_config.enable_verification=True \
    --num_updates=200 \
    --env_config.treat_face_cards_as_10=True \
    --env_config.target_points=24 \
    --run_name=gp_language_training \
    --num_steps=128 \
    --model_path=$CKPT \
    --save_ckpt=True \
    --save_every=$SAVE_EVERY \
    --seed=0
