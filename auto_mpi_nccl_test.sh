#!/bin/bash

# 定义变量 size 的值
sizes=("64" "128" "256" "512" "1K" "2K" "4K" "8K" "16K" "32K" "64K" "128K" "256K" "512K" "1M" "2M" "4M" "8M" "16M" "32M" "64M" "128M" "256M" "512M" "1G")

# 输出结果重定向到 result.txt
exec > result.txt

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL

# 循环 size
for size in "${sizes[@]}"
do
  # 执行命令
  mpirun -np 4 -f 4node.txt /data/home/panlichen/zrk/work2/nccl-tests/build/all_reduce_perf -f 2 -t 8 -g 1 -n1 -w 0 -b $size -e $size
  # 休眠 1 秒
  sleep 1
done