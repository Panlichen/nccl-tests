clear

export MY_NUM_DEV=$1

export NCCL_IGNORE_DISABLED_P2P=1
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
# export NCCL_MAX_NCHANNELS=1
# export NCCL_MIN_NCHANNELS=1
# export NCCL_NTHREADS=64

export NCCL_DEBUG=WARN

mpirun -np 2 -f machinefile /home/panlichen/work2/mpi/nccl-tests/build/all_reduce_perf -b $1 -e $1 -f 2 -t 1 -g 1 -n 1 -w 0 -c 0 > /home/panlichen/work2/ofccl/log/nccl.log 2>&1