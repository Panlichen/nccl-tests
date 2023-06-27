#!/bin/bash

# 定义变量 size 的值
sizes=("64" "128" "256" "512" "1K" "2K" "4K" "8K" "16K" "32K" "64K" "128K" "256K" "512K" "1M" "2M" "4M" "8M" "16M" "32M" "64M" "128M" "256M" "512M" "1G")

# # 输出结果重定向到 result.txt
# exec > result.txt

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL

export PATH=/data/home/panlichen/zrk/mpi/bin:$PATH
export LD_LIBRARY_PATH=/data/home/panlichen/zrk/mpi/lib:/data/home/panlichen/zrk/work2/ofccl/build/lib

# 循环 iter
for iter in {0..5}
do
  # 循环 size
  for size in "${sizes[@]}"
  do
    # 构建命令
    cmd="mpirun -np 4 -f 4node.txt /data/home/panlichen/zrk/work2/nccl-tests/build/all_reduce_perf -f 2 -t 8 -g 1 -n 5 -w 2 -c 0 -b $size -e $size"

    # 打印要执行的命令
    echo "Running command: $cmd"

    # 执行命令并将输出重定向到 iter 对应的文件
    eval "$cmd" | tee -a "nccl_all_reduce$iter.txt"

    # 休眠 10 秒
    sleep 10
  done
  sleep 30
done