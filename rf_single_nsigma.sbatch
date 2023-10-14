#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=rf_nsigma_training            # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=rf_single_nsigma-output.log
#SBATCH --cpus-per-task=4           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                # Number of GPU's
#SBATCH --time=12:00:00              # Max execution time

# Activate your Anaconda environment
# conda activate master_thesis_env

# Run your Python script
cd /home/lbissot/master_thesis/
python rf_single_nsigma.py 