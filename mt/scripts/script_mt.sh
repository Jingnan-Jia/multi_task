#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=6
#SBATCH -t 7-00:00:00
#SBATCH --mem-per-gpu=90G
#SBATCH -e results/slurmlogs/slurm-%j.err
#SBATCH -o results/slurmlogs/slurm-%j.out
#SBATCH --mail-type=end
#SBATCH --mail-user=jiajingnan2222@gmail.com

eval $(conda shell.bash hook)
conda activate py38

job_id=$SLURM_JOB_ID
slurm_dir=results/slurmlogs

scontrol write batch_script ${job_id} ${slurm_dir}/slurm-${job_id}_args.sh
cp ../mymodules/set_args_mtnet.py ${slurm_dir}/slurm-${job_id}_set_args_mtnet.py  # backup setting


time=$(date +%y_%m_%d_%H_%M_%S)  # there must not be any space before and after =

export PYTHONPATH="${PYTHONPATH}:/data/jjia/jjnutils/jjnutils"

idx=0; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u train_mtnet.py 2>${slurm_dir}/slurm-${job_id}_$idx.err 1>${slurm_dir}/slurm-${job_id}_$idx.out --net_names="net_airway_itgt-net_recon" --mode="train" --main_net_name="net_airway_itgt" --ad_lr=0 --ratio_norm_gradients=0 &

wait





