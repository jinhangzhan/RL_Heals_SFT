#!/bin/bash

#SBATCH --job-name=ACM
#SBATCH --mail-user=xxx@email.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1                                 # Total number of nodes requested
#SBATCH --ntasks-per-node=1                       # Total number of task requested
#SBATCH --cpus-per-task=8                        # Total number of cores requested
#SBATCH --mem=192G
#SBATCH -t 24:00:00                          # Time limit (hh:mm:ss)
#SBATCH --gpus-per-node=1   
#SBATCH --output=./slurm_logs/eval_%A_%a.out
#SBATCH --output=./slurm_logs/eval_%A_%a.err

module load cuda/12.2.2/cudnn/8.9
export CUDA_HOME=/cvmfs/ai.mila.quebec/apps/arch/common/cuda/12.2.2
export PYTHONPATH=/home/mila/l/luansito/kim/SFTvsRL:$PYTHONPATH

export WANDB_API_KEY="wandb"
CONDA_DIR="/condapath/"
if [ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
    conda activate SFTvsRL3.13
    echo "Activated conda environment SFTvsRL3.13."
else
    echo "Conda activation script not found. Exiting."
    exit 1
fi


python raw_metrics_angle_acm.py 


