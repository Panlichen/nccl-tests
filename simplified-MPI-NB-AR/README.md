```shell
/data/home/panlichen/mvapich/mvapich-build/bin/mpicc -o osu_iallreduce.out osu_iallreduce.c -I/usr/local/cuda/include -I/data/home/panlichen/mvapich/mvapich-build/include -L/usr/local/cuda/lib64 -L/data/home/panlichen/mvapich/mvapich-build/lib -lcudart -lcuda -lmpi

export LD_LIBRARY_PATH=/data/home/panlichen/mvapich/mvapich-build/lib

/data/home/panlichen/mvapich/mvapich-build/bin/mpiexec -n 8 -f 4node.txt -env MV2_SMP_USE_CMA=0 -env MV2_USE_CUDA=1 -env MV2_HOMOGENEOUS_CLUSTER=1 /data/home/panlichen/mvapich/mvapich-build/get_local_rank /data/home/panlichen/zrk/work2/nccl-tests/simplified-MPI-NB-AR/osu_iallreduce.out


/data/home/panlichen/mvapich/mvapich-build/bin/mpiexec -n 8 -f 4node.txt -env MV2_SMP_USE_CMA=0 -env MV2_USE_CUDA=1 -env MV2_HOMOGENEOUS_CLUSTER=1 /data/home/panlichen/mvapich/mvapich-build/get_local_rank /data/home/panlichen/zrk/work2/nccl-tests/simplified-MPI-NB-AR/osu_iallreduce.out 262144

/data/home/panlichen/mvapich/mvapich-build/bin/mpiexec -n 32 -f 4node.txt -env MV2_SMP_USE_CMA=0 -env MV2_USE_CUDA=1 -env MV2_HOMOGENEOUS_CLUSTER=1 /data/home/panlichen/mvapich/mvapich-build/get_local_rank /data/home/panlichen/zrk/work2/nccl-tests/simplified-MPI-NB-AR/osu_iallreduce.out 262144
```
