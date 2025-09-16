export DEEPSPEED_LOG_LEVEL=trace
export DEEPSPEED_LOGGING=1


module load cuda/12.2
module load cudnn/8.9.5.29
export CUDA_HOME=/path/Core/cudacore/12.2.2


conda activate SFTvsRL3.13
echo "Activated conda environment SFTvsRL3.13."

export PYTHONPATH=/path/RL_Heals_SFT/sft/src:$PYTHONPATH

#MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
MODEL_NAME="/path/data/Llama-3.2-11B-Vision-Instruct"

DATA_JSON="/path/data/SFTvsRL_Data/SFT_Data/virl-l/data.json"
IMAGE_FOLDER="./"
OUTPUT_FOLDER="/path/data/train_ckpt/virl_l_sft"
LR=1e-6
EPOCH=1
LOG_FILE="train-for-virl-l.log"

deepspeed  ../src/training/train.py \
    --deepspeed zero2_offload.json \
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
    --save_steps 50 \
    --save_total_limit 100 \
    --evaluation_strategy steps \
    --eval_steps 5 \
    --dataloader_num_workers 4 \
    --save_only_model True 2>&1 | tee ${LOG_FILE}
