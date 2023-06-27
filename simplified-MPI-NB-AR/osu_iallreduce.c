#include <mpi.h>

#include <assert.h>
#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include "cuda.h"
#include "cuda_runtime.h"

CUcontext cuContext;

int omb_get_local_rank()
{
    char *str = NULL;
    int local_rank = -1;

    if ((str = getenv("MV2_COMM_WORLD_LOCAL_RANK")) != NULL) {
        local_rank = atoi(str);
    } else if ((str = getenv("OMPI_COMM_WORLD_LOCAL_RANK")) != NULL) {
        local_rank = atoi(str);
    } else if ((str = getenv("MPI_LOCALRANKID")) != NULL) {
        local_rank = atoi(str);
    } else if ((str = getenv("LOCAL_RANK")) != NULL) {
        local_rank = atoi(str);
    } else {
        fprintf(stderr, "Warning: OMB could not identify the local rank of the process.\n");
        fprintf(stderr, "         This can lead to multiple processes using the same GPU.\n");
        fprintf(stderr, "         Please use the get_local_rank script in the OMB repo for this.\n");
    }

    return local_rank;
}

int init_accel (void)
{
    CUresult curesult = CUDA_SUCCESS;
    CUdevice cuDevice;
    int local_rank = -1, dev_count = 0;
    int dev_id = 0;

    local_rank = omb_get_local_rank();

    if (local_rank >= 0) {
        cudaGetDeviceCount(&dev_count);
        dev_id = local_rank % dev_count;
    }
    cudaSetDevice(dev_id);

    // printf("local_rank: %d, dev_id: %d\n", local_rank, dev_id);

    curesult = cuInit(0);
    if (curesult != CUDA_SUCCESS) {
        return 1;
    }

    curesult = cuDeviceGet(&cuDevice, dev_id);
    if (curesult != CUDA_SUCCESS) {
        return 1;
    }

    curesult = cuDevicePrimaryCtxRetain(&cuContext, cuDevice);
    if (curesult != CUDA_SUCCESS) {
        return 1;
    }

    return 0;
}


int cleanup_accel (void)
{
    CUresult curesult = CUDA_SUCCESS;

    /* Reset the device to release all resources */
    cudaDeviceReset();

    return 0;
}

int main(int argc, char *argv[]) {

    int i = 0, j, rank, size;
    if (argc < 2) {
      size = 1024;
    } else {
      size = atoi(argv[1]);
    }
    int numprocs;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0;
    double tcomp = 0.0, tcomp_total = 0.0, latency_in_secs = 0.0;
    double test_time = 0.0, test_total = 0.0;
    double timer = 0.0;
    int errors = 0, local_errors = 0;
    double wait_time = 0.0, init_time = 0.0;
    double init_total = 0.0, wait_total = 0.0;
    double avg_time = 0.0, max_time = 0.0, min_time = 0.0;

    char *sendbuf = NULL;
    char *recvbuf = NULL;

    init_accel();
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Request request;
    MPI_Status status;


    cudaMalloc((void**)&sendbuf, size);
    cudaMemset(sendbuf, 1, size);
    cudaMalloc((void**)&recvbuf, size);
    cudaMemset(recvbuf, 0, size);

    // printf("rank: %d, size: %d, sendbuf @ %p, recvbuf @ %p\n", rank, size, sendbuf, recvbuf);

    MPI_Barrier(MPI_COMM_WORLD);

    // for (int i = 0; i < 2; i++) {
    //     MPI_Iallreduce(sendbuf, recvbuf, size / sizeof(float),
    //                 MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD,
    //                 &request);
    //     MPI_Wait(&request,&status);
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }

    
    for (int i = 0; i < 12; i++) {
      t_start = MPI_Wtime();
      MPI_Iallreduce(sendbuf, recvbuf, size / sizeof(float),
                  MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD,
                  &request);
      MPI_Wait(&request,&status);

      t_stop = MPI_Wtime();

      MPI_Barrier(MPI_COMM_WORLD);

      if (i >= 2) {
        timer += t_stop - t_start;
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    latency = (timer * 1e6) / 10;

    MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    avg_time = avg_time/numprocs;

    // 因为上边又跑了MPI_Reduce
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        // printf("min_time: %lf\n", min_time);
        // printf("max_time: %lf\n", max_time);
        printf("%d       %lf\n", size, avg_time);
    }

    cudaFree(sendbuf);
    cudaFree(recvbuf);

    MPI_Finalize();

    cleanup_accel();

    return 0;
}
