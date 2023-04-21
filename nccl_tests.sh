clear

export MY_NUM_DEV=$1

# export LD_LIBRARY_PATH=/home/panlichen/work2/ofccl/build/lib
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
# export NCCL_MAX_NCHANNELS=1
# export NCCL_MIN_NCHANNELS=1
# export NCCL_NTHREADS=64
rm -rf /home/panlichen/work2/ofccl/log
mkdir -p /home/panlichen/work2/ofccl/log/nsys

if [ -z $BINARY ];then
    BINARY="DEBUG"
    # BINARY="MS"
    # BINARY="PERF"
fi

FUNC=$2

if [ "$FUNC" == "AR" ]; then
    target="./build/all_reduce_perf"
elif [ "$FUNC" == "AG" ]; then
    target="./build/all_gather_perf"
elif [ "$FUNC" == "RS" ]; then
    target="./build/reduce_scatter_perf"
elif [ "$FUNC" == "R" ]; then
    target="./build/reduce_perf"
elif [ "$FUNC" == "B" ]; then
    target="./build/broadcast_perf"
fi


if [ "$BINARY" == "DEBUG" ];then
    if [ $MY_NUM_DEV = 4 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3
    fi
    export NITER=5
    export NBYTES=$3
    export WARMITER=2
    export MITER=1
    export CHECK=0
elif [ "$BINARY" == "PERF" ];then
    if [ $MY_NUM_DEV = 4 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,4,5
    fi
    export NITER=4
    export NBYTES=$3
    export WARMITER=2
    export MITER=4
    export CHECK=0
elif [ "$BINARY" == "MS" ];then
    if [ $MY_NUM_DEV = 4 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,4,5
    fi
    # export NITER=200
    # export SHOW_ALL_PREPARED_COLL=1
    # export WARMITER=0
    # export NBYTES=$3
    # export MITER=4
fi

export NSYS_FILE="nccl_"$MY_NUM_DEV"card_"$3
export NCU_FILE="nccl_ncu_"$MY_NUM_DEV"card_"$3

if [ -z $RUN_TYPE ];then
    RUN_TYPE="PURE"
    # RUN_TYPE="GDB"
    # RUN_TYPE="NSYS"
    # RUN_TYPE="NCU"
fi

if [ "$RUN_TYPE" == "PURE" ];then
    cmd="$target -b $NBYTES -e $NBYTES -f 2 -t $MY_NUM_DEV -g 1 -n $NITER -w $WARMITER -c $CHECK -m $MITER"
elif [ "$RUN_TYPE" == "GDB" ];then
    cmd="cuda-gdb $target"
    # set args -b 8M -e 8M -f 2 -t 2 -g 1 -n 1 -w 0 -c 0
elif [ "$RUN_TYPE" == "NSYS" ];then
    cmd="nsys profile -f true --trace=cuda,cudnn,cublas,osrt,nvtx -o /home/panlichen/work2/ofccl/log/nsys/$NSYS_FILE $target -b $NBYTES -e $NBYTES -f 2 -t $MY_NUM_DEV -g 1 -n $NITER -w $WARMITER -c $CHECK -m $MITER"
elif [ "$RUN_TYPE" == "NCU" ];then
    # cmd="ncu --nvtx -f -o /home/panlichen/work2/ofccl/log/nsys/$NCU_FILE $target -b $NBYTES -e $NBYTES -f 2 -t $MY_NUM_DEV -g 1 -n $NITER -w $WARMITER -c $CHECK -m $MITER"
    cmd="ncu $target -b $NBYTES -e $NBYTES -f 2 -t $MY_NUM_DEV -g 1 -n $NITER -w $WARMITER -c $CHECK -m $MITER"
fi

echo cmd=$cmd
$cmd #> /home/panlichen/work2/ofccl/nccl.log

