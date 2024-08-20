#!/bin/sh

# A Slurm script that will run $CMD on the computational node
# It sets all required variables to run torch/huggingface script

#SBATCH --job-name=unnamed
#SBATCH --output=results/slurm_logs/unnamed.log
#SBATCH --error=results/slurm_logs/unnamed.log
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# conda
. "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate asr_research

# cuda
export PATH=/usr/local/cuda-11/bin${PATH:+:${PATH}}

# HF cache path and offline mode
export MPLCONFIGDIR=/userspace/$USER/.matplotlib
export HF_HOME=/userspace/$USER/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# exec
echo $CMD
$CMD
