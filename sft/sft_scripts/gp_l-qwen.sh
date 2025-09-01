#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=sft-qwen-4
#SBATCH --mail-user=xxx@email.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4                   
#SBATCH --mem=500000M          
#SBATCH --time=06:00:00 
#SBATCH --gpus-per-node=4                        
#SBATCH --output=sft_%A_%a.out
#SBATCH --error=sft_%A_%a.err


module load cuda/12.2
module load cudnn/8.9.5.29
export CUDA_HOME=/path/cudacore/12.2.2

CONDA_DIR="conda"
source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate SFTvsRL3.13
echo "Activated conda environment SFTvsRL3.13."

unset NCCL_DEBUG
unset NCCL_DEBUG_SUBSYS
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((10000 + RANDOM % 20000))

HOSTFILE=/path/RL_Heals_SFT/sft/sft_scripts/hostfile_${SLURM_JOB_ID}
scontrol show hostnames "$SLURM_NODELIST" | awk '{print $1" slots=4"}' > "$HOSTFILE"
export ACCELERATE_HOST_FILE=$HOSTFILE
HOSTFILE_OPT="--hostfile $HOSTFILE"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export PYTHONPATH=/path/RL_Heals_SFT/sft/src:$PYTHONPATH
MODEL_NAME="/path/data/Qwen2.5-7B-Instruct"
DATA_JSON="/path/data/SFTvsRL_Data/SFT_Data/gp-l/data.json"
IMAGE_FOLDER="./"
OUTPUT_FOLDER="/path/data/train_ckpt/gp_l_sft-slight-qwen"
NVCC_ROOT=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.2.2/bin/nvcc

LR=1e-6
EPOCH=1

LOG_FILE="train-for-gp-l-slight-qwen-0604.log"

hosts=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
echo hosts
for h in "${hosts[@]}"; do
  ssh-copy-id -i ~/.ssh/id_ed25519.pub $h
done

accelerate launch \
  --num_processes 4 \
  --num_machines $SLURM_JOB_NUM_NODES \
  --mixed_precision bf16 \
  --main_process_port $MASTER_PORT \
  --main_process_ip $MASTER_ADDR \
  --use_deepspeed \
  --deepspeed_config_file zero3_offload.json \
  --deepspeed_multinode_launcher pdsh \
  --deepspeed_hostfile "$HOSTFILE" \
  -m training.train_qwen \
    --model_id $MODEL_NAME \
    --data_path $DATA_JSON \
    --image_folder $IMAGE_FOLDER \
    --disable_flash_attn2 True \
    --lora_enable False \
    --tune_img_projector True \
    --freeze_vision_tower False \
    --freeze_llm False \
    --bf16 True \
    --output_dir $OUTPUT_FOLDER \
    --num_train_epochs ${EPOCH} \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${LR} \
    --projector_lr ${LR} \
    --vision_lr ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to none \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 50 \
    --evaluation_strategy steps \
    --eval_steps 5 \
    --dataloader_num_workers 4 \
    --save_only_model True 2>&1 | tee ${LOG_FILE}
