#!/bin/bash

#SBATCH --job-name=svd_recover
#SBATCH --mail-user=xxx@email.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1                                 # Total number of nodes requested
#SBATCH --ntasks-per-node=1                       # Total number of task requested
#SBATCH --cpus-per-task=8                        # Total number of cores requested
#SBATCH --mem=192G
#SBATCH -t 12:00:00                          # Time limit (hh:mm:ss)
#SBATCH --gpus-per-node=1   
#SBATCH --output=./slurm_logs/eval_%A_%a.out
#SBATCH --output=./slurm_logs/eval_%A_%a.err
#SBATCH --array=0-11

module load cuda/12.2.2/cudnn/8.9
export CUDA_HOME=/cuda_home
export PYTHONPATH=/path/RL_Heals_SFT:$PYTHONPATH

export WANDB_API_KEY="wandb"
CONDA_DIR="/path/conda"
if [ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
    conda activate SFTvsRL3.13
    echo "Activated conda environment SFTvsRL3.13."
else
    echo "Conda activation script not found. Exiting."
    exit 1
fi

K_TOP=(512 1024 0  0  512  1024)
K_TAIL=(512 1024 512 1024 0 0)

idx=$SLURM_ARRAY_TASK_ID
kt=${K_TOP[$idx]}
kb=${K_TAIL[$idx]}

echo ">> running pair (k_top=$kt , k_tail=$kb)"

WORKDIR=$SLURM_TMPDIR/svd_$idx
mkdir -p $WORKDIR
cp ~/kim/SFTvsRL/analysis/svd_recover.py $WORKDIR/run.py

python $WORKDIR/run.py --k_top $kt --k_tail $kb


