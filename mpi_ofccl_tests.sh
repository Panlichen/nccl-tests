clear

export MY_NUM_DEV=$1

FUNC=$2

if [ "$FUNC" == "AR" ]; then
    nccl_target="/home/panlichen/work2/mpi/nccl-tests/build/all_reduce_perf"
    ofccl_target="/home/panlichen/work2/mpi/nccl-tests/build/ofccl_all_reduce_perf"
elif [ "$FUNC" == "AG" ]; then
    nccl_target="/home/panlichen/work2/mpi/nccl-tests/build/all_gather_perf"
    ofccl_target="/home/panlichen/work2/mpi/nccl-tests/build/ofccl_all_gather_perf"
elif [ "$FUNC" == "RS" ]; then
    nccl_target="/home/panlichen/work2/mpi/nccl-tests/build/reduce_scatter_perf"
    ofccl_target="/home/panlichen/work2/mpi/nccl-tests/build/ofccl_reduce_scatter_perf"
elif [ "$FUNC" == "R" ]; then
    nccl_target="/home/panlichen/work2/mpi/nccl-tests/build/reduce_perf"
    ofccl_target="/home/panlichen/work2/mpi/nccl-tests/build/ofccl_reduce_perf"
elif [ "$FUNC" == "B" ]; then
    nccl_target="/home/panlichen/work2/mpi/nccl-tests/build/broadcast_perf"
    ofccl_target="/home/panlichen/work2/mpi/nccl-tests/build/ofccl_broadcast_perf"
fi

export NBYTES=$3

export NCCL_IGNORE_DISABLED_P2P=1
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring

# export NCCL_MAX_NCHANNELS=1
# export NCCL_MIN_NCHANNELS=1

# export NCCL_NTHREADS=64


export CHECK=0
export SHOW_ALL_PREPARED_COLL=0

export RECV_SUCCESS_FACTOR=5
export RECV_SUCCESS_THRESHOLD=100000000
export TOLERANT_UNPROGRESSED_CNT=100000
export BASE_CTX_SWITCH_THRESHOLD=80000
export NUM_TRY_TASKQ_HEAD=6
export DEV_TRY_ROUND=10
export CHECK_REMAINING_SQE_INTERVAL=10000
export DEBUG_FILE="/home/panlichen/work2/ofccl/log/oneflow_cpu_rank_"
# rm -rf /home/panlichen/work2/ofccl/log
# mkdir -p /home/panlichen/work2/ofccl/log/nsys

echo RECV_SUCCESS_FACTOR=$RECV_SUCCESS_FACTOR
echo RECV_SUCCESS_THRESHOLD=$RECV_SUCCESS_THRESHOLD
echo TOLERANT_UNPROGRESSED_CNT=$TOLERANT_UNPROGRESSED_CNT
echo BASE_CTX_SWITCH_THRESHOLD=$BASE_CTX_SWITCH_THRESHOLD
echo NUM_TRY_TASKQ_HEAD=$NUM_TRY_TASKQ_HEAD
echo DEV_TRY_ROUND=$DEV_TRY_ROUND
echo CHECK_REMAINING_SQE_INTERVAL=$CHECK_REMAINING_SQE_INTERVAL
echo DEBUG_FILE=$DEBUG_FILE


# mpirun -np 2 -f machinefile $nccl_target -b $NBYTES -e $NBYTES -f 2 -t $1 -g 1 -n 5 -w 2 -c 0 > /home/panlichen/work2/ofccl/nccl.log 2>&1

# export NCCL_DEBUG=INFO

# export NCCL_IB_DISABLE=1

echo $ofccl_target
mpirun -np 2 -f machinefile $ofccl_target -b $NBYTES -e $NBYTES -f 2 -t $1 -g 1 -n 5 -w 2 -c 0 > /home/panlichen/work2/ofccl/ofccl.log 2>&1


# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=1
# mpirun -np 2 $ofccl_target -b $NBYTES -e $NBYTES -f 2 -t $1 -g 1 -n 1 -w 0 -c 0 #> /home/panlichen/work2/ofccl/ofccl.log 2>&1