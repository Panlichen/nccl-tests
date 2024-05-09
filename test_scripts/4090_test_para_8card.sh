#!/bin/bash
#SBATCH --gpus=8
#SBATCH -p gpu_4090
module load cuda/11.8
python 4090_test_para_8card.py