#!/bin/bash
#SBATCH -J train
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH -o ./out/train.out
#SBATCH -e ./out/train.error

echo ${SLURM_JOB_NODELIST}
echo start on $(date)
/hpc/data/home/bme/v-sunyh2/software/anaconda3/envs/yuhang/bin/python ../train.py
echo end on $(date)
