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

#define NUM_DEV_PER_NODE 8
#define NUM_COLLS 8

#define WARM_ITER 5
#define ITER 20

#define MY_NBC_ON 1  // 1: use MPI non-blocking collectives; 0: use MPI blocking collectives
#define USE_CUDA 1  // 1: use CUDA buffer; 0: use host buffer
#define CHAOS_ORDER 1  // 1: use chaos order; 0: use same order

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

    int size_list[NUM_COLLS] = {256, 147456, 2048, 1024, 65536, 147456, 524288, 1048576};

    int numprocs;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0;
    double tcomp = 0.0, tcomp_total = 0.0, latency_in_secs = 0.0;
    double test_time = 0.0, test_total = 0.0;
    double timer = 0.0;
    int errors = 0, local_errors = 0;
    double wait_time = 0.0, init_time = 0.0;
    double init_total = 0.0, wait_total = 0.0;
    double avg_time = 0.0, max_time = 0.0, min_time = 0.0;

    int coll_id_list[NUM_DEV_PER_NODE][NUM_COLLS] = {
        {0, 5, 2, 6, 4, 1, 7, 3}, 
        {1, 0, 3, 2, 5, 4, 6, 7}, 
        {2, 1, 0, 3, 6, 7, 4, 5}, 
        {3, 2, 1, 0, 7, 6, 5, 4}, 
        {4, 3, 5, 7, 0, 2, 1, 6}, 
        {5, 6, 7, 4, 1, 0, 3, 2}, 
        {6, 7, 4, 1, 3, 5, 2, 0}, 
        {7, 4, 6, 5, 2, 3, 0, 1}
    };

    #ifdef USE_CUDA
    init_accel();
    #endif
    
    MPI_Init(&argc, &argv);
    // 后边对时间进行Reduce，使用这个，而不是单独的comm
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    char *sendbufs[NUM_COLLS];
    char *recvbufs[NUM_COLLS];
    MPI_Comm comms[NUM_COLLS];

    for (i = 0; i < NUM_COLLS; i++) {
        MPI_Comm_dup(MPI_COMM_WORLD, &comms[i]);
    }

    // 这里应该没必要重新在复制出来的comm里再算一遍rank和numprocs
    // for (i = 0; i < NUM_COLLS; i++) {
    //     MPI_Comm_rank(comms[i], &rank);
    //     MPI_Comm_size(comms[i], &numprocs);
    // }
    
    MPI_Request requests[NUM_COLLS];
    MPI_Status status_list[NUM_COLLS];

    for (i = 0; i < NUM_COLLS; i++) {
        size = size_list[i];

        #ifdef USE_CUDA
        cudaMalloc((void**)&sendbufs[i], size);
        cudaMemset(sendbufs[i], 1, size);
        cudaMalloc((void**)&recvbufs[i], size);
        cudaMemset(recvbufs[i], 0, size);
        printf("CUDA buffer: rank: %d, coll_id: %d, size: %d, sendbuf @ %p, recvbuf @ %p\n", rank, i, size, sendbufs[i], recvbufs[i]);
        #else
        size_t alignment = sysconf(_SC_PAGESIZE);
        posix_memalign((void**)&sendbufs[i], alignment, size);
        memset(sendbufs[i], 1, size);
        posix_memalign((void**)&recvbufs[i], alignment, size);
        memset(recvbufs[i], 0, size);
        printf("Host buffer: rank: %d, coll_id: %d, size: %d, sendbuf @ %p, recvbuf @ %p\n", rank, i, size, sendbufs[i], recvbufs[i]);
        #endif
        
    }


    for (i = 0; i < NUM_COLLS; i++) {
        MPI_Barrier(comms[i]);
    }

    // comm是和coll一一对应的，进一步和buffer size一一对应

    for (int i = 0; i < WARM_ITER + ITER; i++) {
        for (int j = 0; j < NUM_COLLS; j++) {

            // rank决定了使用哪个集合通信的调用序列，然后依次调用。
            #ifdef CHAOS_ORDER
            int coll_id = coll_id_list[rank % NUM_DEV_PER_NODE][j];
            #else
            int coll_id = j;  // 先让大家用同样的顺序启动
            #endif

            // printf("rank: %d, coll_id: %d, count: %d\n", rank, coll_id, size_list[coll_id]);

            #ifdef MY_NBC_ON
            MPI_Iallreduce(sendbufs[coll_id], recvbufs[coll_id], size_list[coll_id] / sizeof(float),
                        MPI_FLOAT, MPI_SUM, comms[coll_id],
                        &requests[coll_id]);
            
            printf("NBC, rank: %d, coll_id: %d, count: %d in %dth iter\n", rank, coll_id, size_list[coll_id], i);
            #else
            MPI_Allreduce(sendbufs[coll_id], recvbufs[coll_id], size_list[coll_id] / sizeof(float),
                        MPI_FLOAT, MPI_SUM, comms[coll_id]);
            MPI_Barrier(comms[coll_id]);
            
            printf("BLOCKING, rank: %d, coll_id: %d, count: %d in %dth iter\n", rank, coll_id, size_list[coll_id], i);
            #endif
        }
        
        #ifdef MY_NBC_ON
        MPI_Waitall(NUM_COLLS, requests, status_list);
        
        for (int j = 0; j < NUM_COLLS; j++) {
            #ifdef CHAOS_ORDER
            int coll_id = coll_id_list[rank % NUM_DEV_PER_NODE][j];
            #else
            int coll_id = j;  // 先让大家用同样的顺序启动
            #endif
            MPI_Barrier(comms[coll_id]);
        }
        #endif

        printf("rank: %d, colls done for %dth iter!!!!!!!!!!!!!!!!!!\n", rank, i);
    }

    // t_start = MPI_Wtime();

    // MPI_Iallreduce(sendbuf, recvbuf, size / sizeof(float),
    //             MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD,
    //             &request);
    // MPI_Wait(&request,&status);

    // t_stop = MPI_Wtime();

    // MPI_Barrier(MPI_COMM_WORLD);

    // timer += t_stop - t_start;

    // latency = (timer * 1e6);

    MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    avg_time = avg_time/numprocs;

    // 因为上边又跑了MPI_Reduce
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        // printf("min_time: %lf\n", min_time);
        // printf("max_time: %lf\n", max_time);
        printf("avg_time: %lf\n", avg_time);
    }

    for (i = 0; i < NUM_COLLS; i++) {
        #ifdef USE_CUDA
        cudaFree(sendbufs[i]);
        cudaFree(recvbufs[i]);
        #else
        free(sendbufs[i]);
        free(recvbufs[i]);
        #endif
    }
    // printf("rank: %d, cudaFree OK\n", rank);
    
    for (i = 0; i < NUM_COLLS; i++) {
        MPI_Comm_free(&comms[i]);
    }
    // printf("rank: %d, MPI_Comm_free OK\n", rank);

    MPI_Finalize();
    // printf("rank: %d, MPI_Finalize OK\n", rank);

    #ifdef USE_CUDA
    cleanup_accel();
    #endif
    // printf("rank: %d, cleanup_accel OK\n", rank);

    return 0;
}
