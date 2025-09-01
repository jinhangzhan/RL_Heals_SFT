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
OOD=False
FACE10=True
TARGET=24
NUM_TRAJ=234
check_point=800-29
today=$(date +"%Y-%m-%d")

CKPT_NAME="/path"
OUTPUT_FOLDER="logs/gp_l_indist_verify_${VITER}_target_${TARGET}"

PORT=$((RANDOM % 10000 + 1000))

echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Checkpoint: $CKPT_NAME"
echo "Output folder: $OUTPUT_FOLDER"
echo "Rendezvous port: $PORT"

accelerate launch \
    --config_file /path/RL_Heals_SFT/scripts/config_zero2_1gpu.yaml \
    --main_process_port ${PORT} \
    -m evaluation.launcher \
    -f /path/RL_Heals_SFT/evaluation/configs/llama_gp_language-qwen.yaml \
    --model_path=${CKPT_NAME} \
    --output_dir=${OUTPUT_FOLDER}/gp_l_indist.jsonl \
    --prompt_config.enable_verification=${ENABLE} \
    --env_config.target_points=${TARGET} \
    --env_config.verify_iter=${VITER} \
    --env_config.treat_face_cards_as_10=${FACE10} \
    --env_config.ood=${OOD} \
    --num_traj=${NUM_TRAJ} \
  | tee eval-ind-recover-dir-qwen-800-29-${today}-tamia-${check_point}.log


