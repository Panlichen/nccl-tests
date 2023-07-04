clear

echo “USAGE: bash ofccl_tests.sh MY_NUM_DEV FUNC SIZE”

export MY_NUM_DEV=$1

export DEBUG_CC=1
export DEBUG_ENQ=1

unset DEBUG_CC
unset DEBUG_ENQ

export DEBUG_NT=1 # for debug nccl-tests
unset DEBUG_NT

# export NCCL_P2P_DISABLE=1

export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
# export NCCL_MAX_NCHANNELS=1
# export NCCL_MIN_NCHANNELS=1
# export NCCL_NTHREADS=64

export CHECK=0
export SHOW_ALL_PREPARED_COLL=0

export RECV_SUCCESS_FACTOR=5
export RECV_SUCCESS_THRESHOLD=100000000
export TOLERANT_UNPROGRESSED_CNT=10000
export BASE_CTX_SWITCH_THRESHOLD=200000
export NUM_TRY_TASKQ_HEAD=6
export DEV_TRY_ROUND=10
export CHECK_REMAINING_SQE_INTERVAL=10000
export DEBUG_FILE="/home/panlichen/work2/ofccl/log/oneflow_cpu_rank_"
rm -rf /home/panlichen/work2/ofccl/log
mkdir -p /home/panlichen/work2/ofccl/log/nsys

# export ENABLE_VQ=1 # volunteer quit
# export TOLERANT_FAIL_CHECK_SQ_CNT=5000
# export CNT_BEFORE_QUIT=5

echo RECV_SUCCESS_FACTOR=$RECV_SUCCESS_FACTOR
echo RECV_SUCCESS_THRESHOLD=$RECV_SUCCESS_THRESHOLD
echo TOLERANT_UNPROGRESSED_CNT=$TOLERANT_UNPROGRESSED_CNT
echo BASE_CTX_SWITCH_THRESHOLD=$BASE_CTX_SWITCH_THRESHOLD
echo NUM_TRY_TASKQ_HEAD=$NUM_TRY_TASKQ_HEAD
echo DEV_TRY_ROUND=$DEV_TRY_ROUND
echo CHECK_REMAINING_SQE_INTERVAL=$CHECK_REMAINING_SQE_INTERVAL
echo DEBUG_FILE=$DEBUG_FILE

if [ ! -z $ENABLE_VQ ];then
    echo TOLERANT_FAIL_CHECK_SQ_CNT=$TOLERANT_FAIL_CHECK_SQ_CNT
    echo CNT_BEFORE_QUIT=$CNT_BEFORE_QUIT
fi

FUNC=$2
if [ -z $FUNC ]; then
    FUNC="AR"
fi

if [ "$FUNC" == "AR" ]; then
    target="./build/ofccl_all_reduce_perf"
elif [ "$FUNC" == "AG" ]; then
    target="./build/ofccl_all_gather_perf"
elif [ "$FUNC" == "RS" ]; then
    target="./build/ofccl_reduce_scatter_perf"
elif [ "$FUNC" == "R" ]; then
    target="./build/ofccl_reduce_perf"
elif [ "$FUNC" == "B" ]; then
    target="./build/ofccl_broadcast_perf"
fi

echo "FUNC="$FUNC
echo "target="$target

if [ -z $BINARY ];then
    BINARY="DEBUG"
    # BINARY="MS"
    # BINARY="PERF"
    # BINARY="CHAOS"
fi

if [ "$BINARY" == "DEBUG" ];then
    if [ $MY_NUM_DEV = 4 ]; then
        if [ "$CARDNAME" = "ampere" ]; then
            # export CUDA_VISIBLE_DEVICES=0,1,2,3
            export CUDA_VISIBLE_DEVICES=4,5,6,7
        else
            export CUDA_VISIBLE_DEVICES=0,1,4,5
        fi
    fi
    if [ $MY_NUM_DEV = 2 ]; then
        export CUDA_VISIBLE_DEVICES=4,5
    fi
    export NITER=5
    export NBYTES=$3
    export WARMITER=2
    export MITER=1
elif [ "$BINARY" == "PERF" ];then
    if [ $MY_NUM_DEV = 4 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,4,5
    fi
    export NITER=8
    export NBYTES=$3
    export WARMITER=2
    export MITER=1
elif [ "$BINARY" == "MS" ];then
    target="./build/ofccl_all_reduce_ms_perf"
    if [ $MY_NUM_DEV = 4 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,4,5
    fi
    export NITER=200
    export SHOW_ALL_PREPARED_COLL=1
    export WARMITER=0
    export NBYTES=1K
    export MITER=4
    export CHECK=0
    export RECV_SUCCESS_FACTOR=5
    export RECV_SUCCESS_THRESHOLD=2000
    export BASE_CTX_SWITCH_THRESHOLD=100
    export TOLERANT_UNPROGRESSED_CNT=80000
    export NUM_TRY_TASKQ_HEAD=5
elif [ "$BINARY" == "CHAOS" ];then
    target="./build/ofccl_all_reduce_chaos_order_perf"
    if [ $MY_NUM_DEV = 4 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,4,5
    fi
    export NITER=20
    export SHOW_ALL_PREPARED_COLL=1
    export WARMITER=0
    export NBYTES=1K
    export MITER=1
    export CHECK=0
    export RECV_SUCCESS_FACTOR=5
    export RECV_SUCCESS_THRESHOLD=1000
    export TOLERANT_UNPROGRESSED_CNT=10000
    export BASE_CTX_SWITCH_THRESHOLD=200
    export NUM_TRY_TASKQ_HEAD=3
    export DEV_TRY_ROUND=10
    export CHECK_REMAINING_SQE_INTERVAL=10000
fi

export NSYS_FILE="ofccl"
export NCU_FILE="ofccl"

if [ -z $RUN_TYPE ];then
    RUN_TYPE="PURE"
    # RUN_TYPE="GDB"
    # RUN_TYPE="NSYS"
    # RUN_TYPE="NCU"
fi

if [ "$RUN_TYPE" == "PURE" ];then
    cmd="$target -b $NBYTES -e $NBYTES -f 2 -t $MY_NUM_DEV -g 1 -n $NITER -w $WARMITER -c $CHECK -M $MITER" #  -d half
elif [ "$RUN_TYPE" == "GDB" ];then
    cmd="cuda-gdb $target"
    # set args -b 64 -e 64 -f 2 -t 2 -g 1 -n 1 -w 0 -c 0
    # set args -b 512 -e 512 -f 2 -t 4 -g 1 -n 1 -w 0 -c 0
elif [ "$RUN_TYPE" == "NSYS" ];then
    cmd="nsys profile -f true --trace=cuda,cudnn,cublas,osrt,nvtx -o /home/panlichen/work2/ofccl/log/nsys/$NSYS_FILE $target -b $NBYTES -e $NBYTES -f 2 -t $MY_NUM_DEV -g 1 -n $NITER -w $WARMITER -c $CHECK -M $MITER"
elif [ "$RUN_TYPE" == "NCU" ];then
    # cmd="ncu --nvtx -f -o /home/panlichen/work2/ofccl/log/nsys/$NCU_FILE $target -b $NBYTES -e $NBYTES -f 2 -t $MY_NUM_DEV -g 1 -n $NITER -w $WARMITER -c $CHECK -M $MITER"
    cmd="ncu $target -b $NBYTES -e $NBYTES -f 2 -t $MY_NUM_DEV -g 1 -n $NITER -w $WARMITER -c $CHECK -M $MITER"
fi

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=NET

echo cmd=$cmd
$cmd #> /home/panlichen/work2/ofccl/ofccl.log

