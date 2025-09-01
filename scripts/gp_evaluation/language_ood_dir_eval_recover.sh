#!/bin/bash

#SBATCH --job-name=EvalGP
#SBATCH --mail-user=xxx@email.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH -t 3:00:00
#SBATCH --gpus-per-node=1   
#SBATCH --output=./slurm_logs/eval_%A_%a.out
#SBATCH --output=./slurm_logs/eval_%A_%a.err

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
# enable verification
ENABLE=True
# enable rule-ood eval
OOD=True
# choose rule: face card as 11,12,13
FACE10=False
# specify target: 24
TARGET=24

check_point=3600
today=$(date -d "now" +%Y-%m-%d)

top_k=4000
tail_k=4000

NUM_TRAJ=234
CKPT_NAME="/path/checkpoint-${check_point}-restored-vector-dir--${top_k}--${tail_k}"
OUTPUT_FOLDER="logs/gp_l_ood_verify_${VITER}_target_${TARGET}_${today}_${check_point}"
PORT=$((RANDOM % 10000 + 1000))

accelerate launch \
    --config_file /path/RL_Heals_SFT/scripts/config_zero2_1gpu.yaml --main_process_port ${PORT} \
    -m evaluation.launcher -f /path/RL_Heals_SFT/evaluation/configs/llama_gp_language.yaml \
    --model_path=${CKPT_NAME} \
    --output_dir=${OUTPUT_FOLDER}/gp_l_ood.jsonl \
    --prompt_config.enable_verification=${ENABLE} \
    --env_config.target_points=${TARGET} \
    --env_config.verify_iter=${VITER} \
    --env_config.treat_face_cards_as_10=${FACE10} \
    --env_config.ood=${OOD} \
    --num_traj=${NUM_TRAJ} | tee eval-ood-res-dir-${today}-ck-${check_point}-${top_k}-${tail_k}.log
