#!/bin/sh

#SBATCH --job-name=check_pytorch
#SBATCH --output=tmp/output.log
#SBATCH --error=tmp/output.log
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

pwd

# conda
. "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate asr_research

# cuda
export PATH=/usr/local/cuda-11/bin${PATH:+:${PATH}}

# exec
nvidia-smi -L
python -c "import torch; print(torch.version.cuda)"
