#!/bin/bash
#SBATCH --job-name=EvalGP_ood
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --array=0-11
#SBATCH --output=slurm_logs/ood_%A_%a.out
#SBATCH --error =slurm_logs/ood_%A_%a.err

module load cuda/12.2.2/cudnn/8.9
source ~/miniconda3/etc/profile.d/conda.sh
conda activate SFTvsRL3.13
export PYTHONPATH=/home/mila/l/luansito/kim/SFTvsRL:$PYTHONPATH

K_TOP=(512 1024 0  0  512  1024)
K_TAIL=(512 1024 512 1024 0 0)
idx=$SLURM_ARRAY_TASK_ID
top_k=${K_TOP[$idx]}
tail_k=${K_TAIL[$idx]}

VITER=5
ENABLE=True
OOD=True
FACE10=False
TARGET=24
NUM_TRAJ=234
checkpoint=3600
today=$(date +%Y-%m-%d)

CKPT_NAME="/path/checkpoint-${checkpoint}-restored-vector--${top_k}--${tail_k}"
OUTPUT_FOLDER="logs/gp_l_ood_verify_${VITER}_target_${TARGET}_${today}_${top_k}_${tail_k}"

PORT=$((RANDOM % 10000 + 1000))

accelerate launch \
  --config_file /path/RL_Heals_SFT/scripts/config_zero2_1gpu.yaml \
  --main_process_port ${PORT} \
  -m evaluation.launcher \
  -f /path/RL_Heals_SFT/evaluation/configs/llama_gp_language.yaml \
  --model_path=${CKPT_NAME} \
  --output_dir=${OUTPUT_FOLDER}/gp_l_ood.jsonl \
  --prompt_config.enable_verification=${ENABLE} \
  --env_config.target_points=${TARGET} \
  --env_config.verify_iter=${VITER} \
  --env_config.treat_face_cards_as_10=${FACE10} \
  --env_config.ood=${OOD} \
  --num_traj=${NUM_TRAJ} | tee eval-ood-dir-${today}-ck-${checkpoint}-${top_k}-${tail_k}.log
