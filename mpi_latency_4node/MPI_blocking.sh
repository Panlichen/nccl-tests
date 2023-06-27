#!/bin/bash

# 定义变量 size 的值
# sizes=("64", "128", "256", "512", "1024", "2048", "4096", "8192", "16384", "32768", "65536", "131072", "262144", "524288", "1048576", "2097152", "4194304", "8388608", "16777216", "33554432", "67108864", "134217728", "268435456", "536870912", "1073741824")
sizes=("64", "1073741824")

# # 输出结果重定向到 result.txt
# exec > result.txt

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL

export LD_LIBRARY_PATH=/data/home/panlichen/mvapich/mvapich-build/lib

# 循环 iter
for iter in {0..5}
do
  # 循环 size
  for size in "${sizes[@]}"
  do
    # 构建命令
    cmd="/data/home/panlichen/mvapich/mvapich-build/bin/mpiexec -n 32 -f 4node.txt -env MV2_SMP_USE_CMA=0 -env MV2_USE_CUDA=1 -env MV2_HOMOGENEOUS_CLUSTER=1 /data/home/panlichen/mvapich/mvapich-build/get_local_rank /data/home/panlichen/mvapich/mvapich-build/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce -d cuda -x 2 -i 10 -m $size:$size -M 1273741824"

    # 打印要执行的命令
    echo "Running command: $cmd"

    # 执行命令并将输出重定向到 iter 对应的文件
    eval "$cmd" | tee -a "MPI_blocking_all_reduce-1G$iter.txt"

    # 休眠 10 秒
    sleep 10
  done
  sleep 30
done