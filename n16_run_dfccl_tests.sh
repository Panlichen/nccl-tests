#!/bin/bash
#SBATCH --gpus=2
#SBATCH -p gpu



module load cuda/11.8

source n16_env.sh

GPU_NUM=$(awk -F= '/^#SBATCH[[:space:]]+--gpus=/ {print $2; exit}' "${BASH_SOURCE[0]}")
: "${GPU_NUM:=1}"

bash ofccl_tests.sh "$GPU_NUM" AR 4m
