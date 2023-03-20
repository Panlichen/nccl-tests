clear

export MY_NUM_DEV=$1
export NBYTES=$2

export NCCL_IGNORE_DISABLED_P2P=1
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring

# export NCCL_MAX_NCHANNELS=1
# export NCCL_MIN_NCHANNELS=1

# export NCCL_NTHREADS=64

# export NCCL_DEBUG=INFO

# export NCCL_IB_DISABLE=1

mpirun -np 2 -f machinefile /home/panlichen/work2/mpi/nccl-tests/build/all_reduce_perf -b $2 -e $2 -f 2 -t $1 -g 1 -n 1 -w 0 -c 0 > /home/panlichen/work2/ofccl/nccl.log 2>&1


# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=1
# mpirun -np 2 /home/panlichen/work2/mpi/nccl-tests/build/all_reduce_perf -b $2 -e $2 -f 2 -t $1 -g 1 -n 1 -w 0 -c 0 > /home/panlichen/work2/ofccl/nccl.log 2>&1
