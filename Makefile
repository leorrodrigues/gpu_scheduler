####################
#        SCHEDULER      #
####################

#Programs
CC = gcc
CXX = g++
NVCC = nvcc
PGCXX = pgc++
LD = nvcc
RM = rm -f

#Folders
MULTICRITERIA_PATH = gpuScheduler/multicriteria/
DATACENTER_PATH = gpuScheduler/datacenter/
CLUSTERING_PATH = gpuScheduler/clustering/
VNE_PATH = gpuScheduler/thirdparty/vnegpu/
TOPOLOGY_PATH = gpuScheduler/topology/
GPUSCHEDULER_PATH= gpuScheduler/
BUILD_GPUSCHEDULER = gpuScheduler/build/
BUILD_SIMULATOR = simulator/build/
BUILD_MAIN = ./
TARGET_NAME = main
TARGET = $(BUIILD_MAIN)$(TARGET_NAME)

#Options
BOOST_FLAGS = -DBOOST_LOG_DYN_LINK
OBJ = .o

#If debug was set

ifeq ($(DBG),1)
	DEBUG_CXX = -g
	DEBUG_NVCC = -g
endif

#Flags
#cpp
CXXFLAGS = $(DEBUG_CXX) $(BOOSTFLAGS) -std=c++17 -Wall -D_GLIBCXX_ASSERTIONS -D_FORTIFY_SOURCE=2 -fasynchronous-unwind-tables -fstack-clash-protection -fstack-protector-strong  -pipe -Werror=format-security -fconcepts -Ofast

#nvcc
NVCCFLAGS = $(DEBUG_NVCC) -std=c++14 -Xptxas  -O3 -use_fast_math --gpu-architecture=compute_30 --gpu-code=sm_30,compute_30 -lineinfo

#PGC++
PGCXXFLAGS = -fast -ta=tesla:cc60

#nve
GPUSCHEDULER_FLAG = -I "gpuScheduler"
THIRDPARTY_FLAGS = -I "gpuScheduler/thirdparty/"

LDFLAGS = -lcublas -lboost_program_options

#Generate the object file
CXX_OBJ = -c
NVCC_OBJ = -dc

#Generate executable file
COUT = -o
NVOUT = -o

#Program
#WARNING DON'T INCLUDE HERE THE HEADER FILES.
#PUT ONLY THE FILE NAMES OF THE .CPP OR .CU
#WITHOUT THE EXTENSION.
.PRECIOUS: %$(OBJ)

#Cluster module
CLUSTERING_FILES  := mclInterface

LIST_CLUSTERING := $(foreach file, $(CLUSTERING_FILES), $(BUILD_GPUSCHEDULER)$(file)$(AUX))

LIST_CLUSTERING_OBJ := $(foreach file, $(CLUSTERING_FILES_), $(BUILD_GPUSCHEDULER)$(file)$(OBJ))

LIST_CLUSTERING_DEP := $(foreach file, $(CLUSTERING_FILES), $(BUILD_GPUSCHEDULER)$(file).d)

#Multicriteria module
MULTICRITERIA_FILES := ahp

LIST_MULTICRITERIA := $(foreach file,$(MULTICRITERIA_FILES), $(BUILD_GPUSCHEDULER)$(file)$(AUX))

LIST_MULTICRITERIA_OBJ := $(foreach file,$(MULTICRITERIA_FILES), $(BUILD_GPUSCHEDULER)$(file)$(OBJ))

LIST_MULTICRITERIA_DEP := $(foreach file, $(MULTICRITERIA_FILES), $(BUILD_GPUSCHEDULER)$(file).d)

#GPUSCHEDULER MODULE
GPUSCHEDULER_FILES := gpu_scheduler builder

LIST_GPUSCHEDULER := $(foreach file, $(GPUSCHEDULER_FILES), $(BUILD_GPUSCHEDULER)$(file)$(EXE))

LIST_GPUSCHEDULER_OBJ := $(foreach file, $(GPUSCHEDULER_FILES), $(BUILD_GPUSCHEDULER)$(file)$(OBJ))

LIST_GPUSCHEDULER_DEP := $(foreach file, $(GPUSCHEDULER_FILES), $(BUILD_GPUSCHEDULER)$(file).d)

#Add all the defines in nvcc flags
NVCCFLAGS += $(DEFINES)

.PHONY: gpuscheduler scheduler clustering multicriteria json libs

all:

scheduler_nvcc: clustering multicriteria libs json $(LIST_GPUSCHEDULER)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $(BUILD_GPUSCHEDULER)*.o $(NVOUT) $(GPUSCHEDULER_PATH)gpuscheduler.out
	echo $(NVCC)

scheduler_pgi: pgc clustering multicriteria libs json $(LIST_GPUSCHEDULER)
	$(PGCXX) $(PGCXXFLAGS) $(LDFLAGS) $(BUILD_GPUSCHEDULER)*.o -o $(GPUSCHEDULER_PATH)gpuscheduler.out

ifeq ($(MAKECMDGOALS),scheduler)
include $(LIST_CLUSTERING_DEP)
include $(LIST_MULTICRITERIA_DEP)
include $(LIST_GPUSCHEDULER_DEP)
endif


clustering: $(LIST_CLUSTERING)
ifeq ($(MAKECMDGOALS),clustering)
include $(LIST_CLUSTERING_DEP)
endif

multicriteria: $(LIST_MULTICRITERIA)
ifeq ($(MAKECMDGOALS), multicriteria)
include $(LIST_MULTICRITERIA_DEP)
endif

$(BUILD_GPUSCHEDULER)json$(OBJ): $(GPUSCHEDULER_PATH)json.cpp
	$(CXX) $(CXXFLAGS) $(CXX_OBJ) $< $(COUT)"$@";

#Compiling all the objs in the final executable
$(BUILD_GPUSCHEDULER)%$(AUX) : $(BUILD_GPUSCHEDULER)%$(OBJ);

#Compile the multicriteria module
$(BUILD_GPUSCHEDULER)%$(OBJ) : $(MULTICRITERIA_PATH)%.cpp
	$(CXX) $(CXXFLAGS) $(CXX_OBJ) $< $(COUT) $@;

$(BUILD_GPUSCHEDULER)%.d : $(MULTICRITERIA_PATH)%.cpp
	$(CXX) $(CXXFLAGS) -M $< $(COUT) $@;

#Compile the clustering module
$(BUILD_GPUSCHEDULER)%$(OBJ) : $(CLUSTERING_PATH)%.cu
	$(NVCC) $(NVCCFLAGS) $(THIRDPARTY_FLAGS) $(NVCC_OBJ) $< $(NVOUT) $@;

$(BUILD_GPUSCHEDULER)%.d : $(CLUSTERING_PATH)%.cu
	$(NVCC) $(NVCCFLAGS) $(THIRDPARTY_FLAGS) -odir $(BUILD_GPUSCHEDULER) -M $< $(NVOUT) $@;

#Compile GPUSCHEDULER module
$(BUILD_GPUSCHEDULER)%$(OBJ) : $(GPUSCHEDULER_PATH)%.cu
	$(NVCC) $(NVCCFLAGS)  $(THIRDPARTY_FLAGS) $(NVCC_OBJ) $< $(NVOUT) $@;

$(BUILD_GPUSCHEDULER)%.d : $(GPUSCHEDULER_PATH)%.cu
	$(NVCC) $(NVCCFLAGS) $(THIRDPARTY_FLAGS) -odir $(BUILD_GPUSCHEDULER) -M $< $(NVOUT) $@;


libs: $(BUILD_GPUSCHEDULER)pugixml$(OBJ);

$(BUILD_GPUSCHEDULER)pugixml$(OBJ): $(VNE_PATH)libs/pugixml/src/pugixml.cpp
	$(CXX) $(CXXFLAGS) $(CXX_OBJ) $< $(COUT)"$@";

json: $(BUILD_GPUSCHEDULER)json$(OBJ);

clean:
	rm $(BUILD_GPUSCHEDULER)*.o $(BUILD_GPUSCHEDULER)*.d;
