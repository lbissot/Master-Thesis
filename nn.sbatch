#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=nn_nsigma_training            # Name of the job 
#SBATCH --export=ALL                # Export all environment variables

# Generate the current date and time
current_date=$(date +\%Y\%m\%d\%H\%M\%S)

# Set the output file name with the date
#SBATCH --output=nn_nsigma-output-${current_date}.log

#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=2:00:00              # Max execution time

# Activate your Anaconda environment
conda activate master_thesis_env

# Run your Python script
cd /home/lbissot/master_thesis/
python NN_cluster_file.py --epochs 500 --learning_rate 0.01 --decay_rate 0.1 --batch_size 124 --n_obs_train 500 --hidden_size 256 --n_hidden_layers 10