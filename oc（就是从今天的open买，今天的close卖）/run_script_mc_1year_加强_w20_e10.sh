#!/bin/bash -l
#$ -l mem_per_core=8G             # 每核内存 8G
#$ -l gpus=1                       # 请求一个 GPU
#$ -pe omp 4                       # 请求 4 个 OpenMP 线程
module load python3/3.8.10
module load tensorflow/2.11.0
module load cuda/11.2
module load cudnn/8.1.1

# 运行你的 Python 脚本
python /usr3/graduate/xz0224/oc（就是从今天的open买，今天的close卖）/mc_1year_w20_e10.py
