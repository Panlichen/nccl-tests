clear

export MY_NUM_DEV=$1

export DEBUG_CC=1
export DEBUG_ENQ=1

unset DEBUG_CC
unset DEBUG_ENQ

export DEBUG_NT=1
unset DEBUG_NT

cd /home/panlichen/work2/nccl-tests
export LD_LIBRARY_PATH=/home/panlichen/work2/ofccl/build/lib
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
# export NCCL_MAX_NCHANNELS=1
# export NCCL_MIN_NCHANNELS=1
# export NCCL_NTHREADS=64

export CHECK=0
export SHOW_ALL_PREPARED_COLL=0

export RECV_SUCCESS_FACTOR=5
export RECV_SUCCESS_THRESHOLD=10000
export TOLERANT_UNPROGRESSED_CNT=10000
export BASE_CTX_SWITCH_THRESHOLD=8000
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

if [ -z $BINARY ];then
    BINARY="DEBUG"
    # BINARY="MS"
    # BINARY="PERF"
    # BINARY="CHAOS"
fi

if [ "$BINARY" == "DEBUG" ];then
    if [ $MY_NUM_DEV = 4 ]; then
        if [ "$CARDNAME" = "ampere" ]; then
            export CUDA_VISIBLE_DEVICES=0,1,2,3
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
    export NBYTES=$3
    export MITER=4
    export CHECK=0
elif [ "$BINARY" == "CHAOS" ];then
    target="./build/ofccl_all_reduce_chaos_order_perf"
    if [ $MY_NUM_DEV = 4 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,4,5
    fi
    export NITER=200
    export SHOW_ALL_PREPARED_COLL=1
    export WARMITER=0
    export NBYTES=$3
    export MITER=1
    export CHECK=0
fi

export NSYS_FILE="ofccl"
export NCU_FILE="ofccl"

if [ -z $RUN_TYPE ];then
    RUN_TYPE="PURE"
    # RUN_TYPE="GDB"
    # RUN_TYPE="NSYS"
    # RUN_TYPE="NCU"
fi

# typedef enum { ncclInt8       = 0, ncclChar       = 0,
#                ncclUint8      = 1,
#                ncclInt32      = 2, ncclInt        = 2,
#                ncclUint32     = 3,
#                ncclInt64      = 4,
#                ncclUint64     = 5,
#                ncclFloat16    = 6, ncclHalf       = 6,
#                ncclFloat32    = 7, ncclFloat      = 7,
#                ncclFloat64    = 8, ncclDouble     = 8,
# #if defined(__CUDA_BF16_TYPES_EXIST__)
#                ncclBfloat16   = 9,
#                ncclNumTypes   = 10
# #else
#                ncclNumTypes   = 9
# #endif
# } ncclDataType_t;

# 用这个：
# const char *test_typenames[ncclNumTypes] = {"int8",
#                                             "uint8",
#                                             "int32",
#                                             "uint32",
#                                             "int64",
#                                             "uint64",
#                                             "half",
#                                             "float",
#                                             "double"
# #if defined(__CUDA_BF16_TYPES_EXIST__) &&                                      \
#     NCCL_VERSION_CODE >= NCCL_VERSION(2, 10, 0)
#                                             ,
#                                             "bfloat16"
# #endif
# };

if [ "$RUN_TYPE" == "PURE" ];then
    cmd="$target -b $NBYTES -e $NBYTES -f 2 -t $MY_NUM_DEV -g 1 -n $NITER -w $WARMITER -c $CHECK -M $MITER" #  -d half
elif [ "$RUN_TYPE" == "GDB" ];then
    cmd="cuda-gdb $target"
    # set args -b 64 -e 64 -f 2 -t 2 -g 1 -n 1 -w 0 -c 0
elif [ "$RUN_TYPE" == "NSYS" ];then
    cmd="nsys profile -f true --trace=cuda,cudnn,cublas,osrt,nvtx -o /home/panlichen/work2/ofccl/log/nsys/$NSYS_FILE $target -b $NBYTES -e $NBYTES -f 2 -t $MY_NUM_DEV -g 1 -n $NITER -w $WARMITER -c $CHECK -M $MITER"
elif [ "$RUN_TYPE" == "NCU" ];then
    # cmd="ncu --nvtx -f -o /home/panlichen/work2/ofccl/log/nsys/$NCU_FILE $target -b $NBYTES -e $NBYTES -f 2 -t $MY_NUM_DEV -g 1 -n $NITER -w $WARMITER -c $CHECK -M $MITER"
    cmd="ncu $target -b $NBYTES -e $NBYTES -f 2 -t $MY_NUM_DEV -g 1 -n $NITER -w $WARMITER -c $CHECK -M $MITER"
fi

echo cmd=$cmd
$cmd #> /home/panlichen/work2/ofccl/log/ofccl.log
