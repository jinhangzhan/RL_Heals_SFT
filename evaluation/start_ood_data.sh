#!/bin/bash

#SBATCH --job-name=OOD_DATA
#SBATCH --mail-user=xxx@email.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500000M
#SBATCH -t 01:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=./slurm_logs/eval_%A_%a.out
#SBATCH --error=./slurm_logs/eval_%A_%a.err
#SBATCH --gres=gpu:h100:4
#SBATCH --export=LD_LIBRARY_PATH=

unset SETUPTOOLS_USE_DISTUTILS
module --force purge
module load StdEnv/2023
module load cuda/12.2
module load cudnn/8.9.5.29

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib64:$LD_LIBRARY_PATH"
export DS_BUILD_OPS=0
export TRITON_DISABLE=1

export C10_DISABLE_LEVEL_ZERO=1
export C10_DISABLE_CPUPOWER=1

unset PYTHONPATH

nvidia-smi || echo "NO GPU!"

echo "=== MODULE LIST ==="
module list
echo "==================="

export CUDA_HOME=/path/cudacore/12.2.2

export WANDB_MODE=offline

export WANDB_API_KEY="wandbKey"

export PYTHONPATH=/path/RL_Heals_SFT:$PYTHONPATH

CONDA_DIR="/path/conda"
if [ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
    conda activate SFTvsRL3.13
    echo "Activated conda environment SFTvsRL3.13"
else
    echo "Conda activation script not found. Exiting."
    exit 1
fi

export PATH=$CONDA_PREFIX/bin:${PATH//:$HOME/.local/bin/}
export PYTHONNOUSERSITE=1 

VITER=5
# enable verification
ENABLE=True
# enable rule-ood eval
OOD=True
# choose rule: face card as 11,12,13
FACE10=False
# specify target: 24
TARGET=24

check_point=6500
today=$(date -d "now" +%Y-%m-%d)

NUM_TRAJ=234
CKPT_NAME="/path/Llama-3.2-11B-Vision-Instruct/"
PORT=$((RANDOM % 10000 + 1000))

OUTPUT_FOLDER="logs/gp_l_ood_verify_${VITER}_target_${TARGET}_${today}_${check_point}"

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
    --num_traj=${NUM_TRAJ} | tee eval-ood-res-${today}-ck-${check_point}.log
