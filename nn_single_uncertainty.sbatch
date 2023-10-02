#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=nn_nsigma_training            # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=nn_single_uncertainty-output.log
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=12:00:00              # Max execution time

# Activate your Anaconda environment
# conda activate master_thesis_env

# Run your Python script
cd /home/lbissot/master_thesis/
python nn_single_uncertainty.py --epochs 1000 --learning_rate 0.005 --decay_rate 0.9 --batch_size 1 --n_obs_train 1000 --hidden_size 512 --n_hidden_layers 8