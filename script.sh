#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH -t 7-00:00:00
#SBATCH --mem-per-gpu=150G
#SBATCH --mail-type=end
#SBATCH --mail-user=jiajingnan2222@gmail.com
#SBATCH -e slurmlogs/slurm-%j.err
#SBATCH -o slurmlogs/slurm-%j.out

eval $(conda shell.bash hook)
conda activate py37

stdbuf -oL python -u train_mtnet.py




