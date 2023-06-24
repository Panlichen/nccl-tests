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
        CUDA_CHECK(cudaGetDeviceCount(&dev_count));
        dev_id = local_rank % dev_count;
    }
    CUDA_CHECK(cudaSetDevice(dev_id));

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
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
