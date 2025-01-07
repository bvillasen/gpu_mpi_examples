
# Compilers
CC  = cc
CXX = CC

ifeq ($(USE_OMPI), 1)
	CC                = ${OMPI_PATH}/bin/mpicc
	CXX               = ${OMPI_PATH}/bin/mpicxx
endif

GPUCXX            = hipcc

CFLAGS_DEBUG      = -g -O0
CFLAGS_OPTIMIZE   = -g -O2

CXXFLAGS_DEBUG    = -g -O0 -std=c++11
CXXFLAGS_OPTIMIZE = -Ofast -std=c++11

GPUFLAGS          = --offload-arch=$(GPU_ARCH)
HIPCONFIG	  = $(shell hipconfig -C)

COMPILE_TYPE = -DUSE_HIP

DFLAGS += $(COMPILE_TYPE)


DIRS     := ../hip_tools src

CFILES   := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.c))
CPPFILES := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.cpp))
GPUFILES := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.cu))

# Build a list of all potential object files so cleaning works properly
CLEAN_OBJS := $(subst .c,.o,$(CFILES)) \
              $(subst .cpp,.o,$(CPPFILES)) \
              $(subst .cu,.o,$(GPUFILES))

OBJS     := $(subst .c,.o,$(CFILES)) \
            $(subst .cpp,.o,$(CPPFILES)) \
            $(subst .cu,.o,$(GPUFILES))

CFLAGS_OPTIMIZE   ?= -Ofast
CXXFLAGS_OPTIMIZE ?= -Ofast -std=c++11
GPUFLAGS_OPTIMIZE ?= -g -O3 -std=c++11
BUILD             ?= OPTIMIZE

CFLAGS            += $(CFLAGS_$(BUILD))
CXXFLAGS          += $(CXXFLAGS_$(BUILD))
GPUFLAGS          += $(GPUFLAGS_$(BUILD))

#-- Add flags and libraries as needed
CFLAGS   += $(DFLAGS) -Isrc -I../hip_tools -I$(MPICH_DIR)/include
CXXFLAGS += $(DFLAGS) -Isrc -I../hip_tools -I$(MPICH_DIR)/include
GPUFLAGS += $(DFLAGS) -Isrc -I../hip_tools 

# LIBS += -L$(MPICH_DIR)/lib -lmpi -L$(CRAY_MPICH_ROOTDIR)/gtl/lib -lmpi_gtl_hsa
LIBS += -L$(MPICH_DIR)/lib -lmpi 

ifeq ($(findstring -DUSE_HIP,$(DFLAGS)),-DUSE_HIP)
	CXXFLAGS += -I$(ROCM_PATH)/include
	GPUFLAGS += -I$(ROCM_PATH)/include
	LIBS += -L$(ROCM_PATH)/lib -lhiprand
	# CXXFLAGS  += $(HIPCONFIG)
	GPUCXX    ?= hipcc
	GPUFLAGS  += -std=c++11 -Wall -ferror-limit=1 -fPIE
	LD        := $(CXX)
	LDFLAGS   := $(CXXFLAGS)
	LIBS      += -L$(ROCM_PATH)/lib -lamdhip64 
endif		

.SUFFIXES: .c .cpp .cu .o

EXEC := gpu_all_reduce_${GPU_TYPE}


$(EXEC):  $(OBJS)
	$(LD) $(LDFLAGS) $(OBJS) -o $(EXEC) $(LIBS)
	eval $(EXTRA_COMMANDS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(GPUCXX) $(GPUFLAGS) -c $< -o $@ 

.PHONY: clean

clean:
	rm -f $(CLEAN_OBJS) $(EXEC)
