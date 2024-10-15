#!/bin/bash -l

#$ -l gpus=1                      # 请求一个 GPU
#$ -pe omp32 32                   # 使用 16 个线程 (OpenMP)
module load python3/3.8.10
module load tensorflow/2.11.0
module load cuda/11.2
module load cudnn/8.1.1

# 运行你的 Python 脚本
python /usr3/graduate/xz0224/oc（就是从今天的open买，今天的close卖）/qqq_1_year.py

