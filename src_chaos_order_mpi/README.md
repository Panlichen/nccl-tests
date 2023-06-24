```shell
/data/home/panlichen/mvapich/mvapich-build/bin/mpicc -o test_mpi test_mpi.c -I/usr/local/cuda/include -I/data/home/panlichen/mvapich/mvapich-build/include -L/usr/local/cuda/lib64 -L/data/home/panlichen/mvapich/mvapich-build/lib -lcudart -lcuda -lmpi

export LD_LIBRARY_PATH=/data/home/panlichen/mvapich/mvapich-build/lib

/data/home/panlichen/mvapich/mvapich-build/bin/mpiexec -n 8 -f local.txt -env MV2_SMP_USE_CMA=0 -env MV2_USE_CUDA=1 /home/panlichen/work2/mpi/nccl-tests/src_chaos_order_mpi/get_local_rank /home/panlichen/work2/mpi/nccl-tests/src_chaos_order_mpi/test_mpi
```
