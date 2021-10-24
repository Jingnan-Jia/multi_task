#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
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
mkdir -p ${slurm_dir}  # create dir if not exist

scontrol write batch_script ${job_id} ${slurm_dir}/slurm-${job_id}_args.sh
cp ../mymodules/set_args_mtnet.py ${slurm_dir}/slurm-${job_id}_set_args_mtnet.py  # backup setting


#time=$(date +%y_%m_%d_%H_%M_%S)  # there must not be any space before and after =

#export PYTHONPATH="${PYTHONPATH}:/data/jjia/jjnutils/jjnutils"

idx=0; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u train_mtnet.py 2>${slurm_dir}/slurm-${job_id}_$idx.err 1>${slurm_dir}/slurm-${job_id}_$idx.out --loss='weighted_dice' --net_names='net_lobe' --main_net_name='net_lobe' --ad_lr=0.0 --ratio_norm_gradients=0.0 --fat=0 &
#idx=1; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u train_mtnet.py 2>${slurm_dir}/slurm-${job_id}_$idx.err 1>${slurm_dir}/slurm-${job_id}_$idx.out --loss='weighted_dice' --net_names='net_lobe-net_vessel-net_recon' --main_net_name='net_lobe' --ad_lr=0.1 --ratio_norm_gradients=0.0 --fat=1 &

wait





