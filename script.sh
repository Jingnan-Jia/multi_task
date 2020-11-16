#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=10
#SBATCH -t 7-00:00:00
#SBATCH --mem-per-gpu=90G
#SBATCH --mail-type=end
#SBATCH --mail-user=jiajingnan2222@gmail.com
#SBATCH -e slurmlogs/slurm-%j.err
#SBATCH -o slurmlogs/slurm-%j.out

eval $(conda shell.bash hook)
conda activate py37
python run_net.py --mode "train"





