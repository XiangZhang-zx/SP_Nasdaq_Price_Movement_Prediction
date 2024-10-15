#!/bin/bash -l

#$ -l gpus=1                      # 请求一个 GPU
#$ -pe omp 4                       # 请求 4 个 OpenMP 线程
module load python3/3.8.10
module load tensorflow/2.11.0
module load cuda/11.2
module load cudnn/8.1.1

# 运行你的 Python 脚本
python /usr3/graduate/xz0224/co/gv_1year.py

