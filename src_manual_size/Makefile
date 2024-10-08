#
# Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

CUDA_HOME ?= /usr/local/cuda
PREFIX ?= /usr/local
VERBOSE ?= 0
DEBUG_NT ?= 0

CUDA_LIB ?= $(CUDA_HOME)/lib64
CUDA_INC ?= $(CUDA_HOME)/include
NVCC = $(CUDA_HOME)/bin/nvcc
CUDARTLIB ?= cudart

CUDA_VERSION = $(strip $(shell which $(NVCC) >/dev/null && $(NVCC) --version | grep release | sed 's/.*release //' | sed 's/\,.*//'))
CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)

# Better define NVCC_GENCODE in your environment to the minimal set
# of archs to reduce compile time.
# ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 11; echo $$?),0)
# NVCC_GENCODE ?= -gencode=arch=compute_60,code=sm_60 \
#                 -gencode=arch=compute_61,code=sm_61 \
#                 -gencode=arch=compute_70,code=sm_70 \
#                 -gencode=arch=compute_80,code=sm_80 \
#                 -gencode=arch=compute_80,code=compute_80
# else
# NVCC_GENCODE ?= -gencode=arch=compute_35,code=sm_35 \
#                 -gencode=arch=compute_50,code=sm_50 \
#                 -gencode=arch=compute_60,code=sm_60 \
#                 -gencode=arch=compute_61,code=sm_61 \
#                 -gencode=arch=compute_70,code=sm_70 \
#                 -gencode=arch=compute_70,code=compute_70
# endif

CUDA_GENCODE_3080   = -gencode=arch=compute_86,code=sm_86
CUDA_GENCODE_ampere   = -gencode=arch=compute_80,code=sm_80
CUDA_GENCODE_2080   = -gencode=arch=compute_75,code=sm_75


CUDA_GENCODE_h800   = -gencode=arch=compute_89,code=sm_89 \
                      -gencode=arch=compute_90,code=sm_90 \
                      -gencode=arch=compute_90a,code=sm_90a

CARDNAME ?= 3080
ifeq ($(CARDNAME), 3080)
NVCC_GENCODE ?= $(CUDA_GENCODE_3080) $(CUDA_PTX_INUSE)
else ifeq ($(CARDNAME), ampere)
NVCC_GENCODE ?= $(CUDA_GENCODE_ampere) $(CUDA_PTX_INUSE)
else ifeq ($(CARDNAME), h800)
NVCC_GENCODE ?= $(CUDA_GENCODE_h800) $(CUDA_PTX_INUSE)
else
NVCC_GENCODE ?= $(CUDA_GENCODE_2080) $(CUDA_PTX_INUSE)
endif
$(info CARDNAME $(CARDNAME))
$(info NVCC_GENCODE $(NVCC_GENCODE))

NVCUFLAGS  := -ccbin $(CXX) $(NVCC_GENCODE) -std=c++11

LDFLAGS    := -L${CUDA_LIB} -lcudart -lrt
NVLDFLAGS  := -L${CUDA_LIB} -l${CUDARTLIB} -lrt

ifeq ($(DEBUG_NT), 0)
NVCUFLAGS += -O3 -g
CXXFLAGS  += -O3 -g
else
NVCUFLAGS += -O0 -G -g
CXXFLAGS  += -O0 -g -ggdb3
endif

ifneq ($(VERBOSE), 0)
NVCUFLAGS += -Xcompiler -Wall,-Wextra,-Wno-unused-parameter
else
.SILENT:
endif

.PHONY: build clean

BUILDDIR ?= ../build
ifneq ($(NCCL_HOME), "")
NVCUFLAGS += -I$(NCCL_HOME)/include/
NVLDFLAGS += -L$(NCCL_HOME)/lib
endif

ifeq ($(MPI), 1)
NVCUFLAGS += -DMPI_SUPPORT -I$(MPI_HOME)/include
NVLDFLAGS += -L$(MPI_HOME)/lib -L$(MPI_HOME)/lib64 -lmpi
endif
ifeq ($(MPI_IBM),1)
NVCUFLAGS += -DMPI_SUPPORT
NVLDFLAGS += -lmpi_ibm
endif
LIBRARIES += nccl
NVLDFLAGS += $(LIBRARIES:%=-l%)

$(info CARDNAME $(NVCUFLAGS))

DST_DIR := $(BUILDDIR)
SRC_FILES := $(wildcard *.cu)
OBJ_FILES := $(SRC_FILES:%.cu=${DST_DIR}/%.o)
BIN_FILES_LIST := ofccl_all_reduce_ms
BIN_FILES := $(BIN_FILES_LIST:%=${DST_DIR}/%_perf)

build: ${BIN_FILES}

clean:
	rm -rf ${DST_DIR}

${DST_DIR}/%.o: %.cu common_ms.h
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) -c $<

${DST_DIR}/%_perf:${DST_DIR}/%.o ${DST_DIR}/common_ms.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}

