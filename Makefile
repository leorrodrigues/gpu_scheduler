####################
#        SCHEDULER      #
####################

$(shell mkdir -p gpuScheduler/build)
$(shell mkdir -p gpuScheduler/build/hierarchy)

#Programs
CXX = g++
NVCC = nvcc
LD = nvcc
RM = rm -f

#Folders
MULTICRITERIA_PATH = gpuScheduler/multicriteria/
DATACENTER_PATH = gpuScheduler/datacenter/
BUILD_GPUSCHEDULER = gpuScheduler/build/
CLUSTERING_PATH = gpuScheduler/clustering/
TASKS_PATH = gpuScheduler/datacenter/tasks/
VNE_PATH = gpuScheduler/thirdparty/vnegpu/
TOPOLOGY_PATH = gpuScheduler/topology/
GPUSCHEDULER_PATH= gpuScheduler/
BUILD_SIMULATOR = simulator/build/
RABBIT_PATH = gpuScheduler/rabbit/
SIMULATOR_PATH = simulator/
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
SO_NAME:= $(shell cat /etc/os-release | grep "ID=" | egrep -v "*_ID_*" | cut -c4-20)

LIBS_PATH :=;

ifeq ($(SO_NAME),arch)
	RABBIT_LIBS_PATH := gpuScheduler/thirdparty/rabbitmq-c/lib64
endif

ifeq ($(SO_NAME),linuxmint)
	RABBIT_LIBS_PATH := gpuScheduler/thirdparty/rabbitmq-c/lib64
endif

ifeq ($(SO_NAME),ubuntu)
	RABBIT_LIBS_PATH := /usr/lib
endif


CXXFLAGS = $(DEBUG_CXX) -std=c++17 -Wall -D_GLIBCXX_ASSERTIONS -D_FORTIFY_SOURCE=2 -fasynchronous-unwind-tables -fstack-protector-strong  -pipe -Werror=format-security -fconcepts -L$(RABBIT_LIBS_PATH) -lrabbitmq -Ofast

CXXFLAGS_W/BOOST = $(DEBUG_CXX) $(BOOSTFLAGS) -std=c++17 -Wall -D_GLIBCXX_ASSERTIONS -D_FORTIFY_SOURCE=2 -fasynchronous-unwind-tables -fstack-protector-strong  -pipe -Werror=format-security -fconcepts -L$(RABBIT_LIBS_PATH) -lrabbitmq -Ofast

#nvcc
NVCCFLAGS = $(DEBUG_NVCC) -std=c++14 -Xptxas  -O3 -use_fast_math -lineinfo

#nve
GPUSCHEDULER_FLAG = -I "gpuScheduler"
THIRDPARTY_FLAGS = -I "gpuScheduler/thirdparty/"

LDFLAGS = -lcublas -lboost_program_options -L$(RABBIT_LIBS_PATH) -lrabbitmq

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

#Cluster module
CLUSTERING_FILES  := mclInterface

LIST_CLUSTERING := $(foreach file, $(CLUSTERING_FILES), $(BUILD_GPUSCHEDULER)$(file)$(AUX))

LIST_CLUSTERING_OBJ := $(foreach file, $(CLUSTERING_FILES_), $(BUILD_GPUSCHEDULER)$(file)$(OBJ))

LIST_CLUSTERING_DEP := $(foreach file, $(CLUSTERING_FILES), $(BUILD_GPUSCHEDULER)$(file).d)

#Multicriteria module
MULTICRITERIA_FILES := ahp ahpg hierarchy/hierarchy hierarchy/hierarchy_resource hierarchy/node hierarchy/edge

LIST_MULTICRITERIA := $(foreach file,$(MULTICRITERIA_FILES), $(BUILD_GPUSCHEDULER)$(file)$(AUX))

LIST_MULTICRITERIA_OBJ := $(foreach file,$(MULTICRITERIA_FILES), $(BUILD_GPUSCHEDULER)$(file)$(OBJ))

LIST_MULTICRITERIA_DEP := $(foreach file, $(MULTICRITERIA_FILES), $(BUILD_GPUSCHEDULER)$(file).d)

#Rabbit module
RABBIT_FILES := common utils

LIST_RABBIT := $(foreach file,$(RABBIT_FILES), $(BUILD_GPUSCHEDULER)$(file)$(AUX))

LIST_RABBIT_OBJ := $(foreach file,$(RABBIT_FILES), $(BUILD_GPUSCHEDULER)$(file)$(OBJ))

LIST_RABBIT_DEP := $(foreach file, $(RABBIT_FILES), $(BUILD_GPUSCHEDULER)$(file).d)

#Tasks module
TASKS_FILES := container virtualMachine

LIST_TASK := $(foreach file,$(TASKS_FILES), $(BUILD_GPUSCHEDULER)$(file)$(AUX))

LIST_TASK_OBJ := $(foreach file,$(TASKS_FILES), $(BUILD_GPUSCHEDULER)$(file)$(OBJ))

LIST_TASK_DEP := $(foreach file, $(TASKS_FILES), $(BUILD_GPUSCHEDULER)$(file).d)


#GPUSCHEDULER MODULE
GPUSCHEDULER_FILES := gpu_scheduler builder

LIST_GPUSCHEDULER := $(foreach file, $(GPUSCHEDULER_FILES), $(BUILD_GPUSCHEDULER)$(file)$(EXE))

LIST_GPUSCHEDULER_OBJ := $(foreach file, $(GPUSCHEDULER_FILES), $(BUILD_GPUSCHEDULER)$(file)$(OBJ))

LIST_GPUSCHEDULER_DEP := $(foreach file, $(GPUSCHEDULER_FILES), $(BUILD_GPUSCHEDULER)$(file).d)

#SIMULATOR MODULE
SIMULATOR_FILES := utils simulatorTasks

LIST_SIMULATOR_OBJ := $(foreach file, $(SIMULATOR_FILES), $(BUILD_SIMULATOR)$(file)$(OBJ))

LIST_SIMULATOR_DEP := $(foreach file, $(SIMULATOR_FILES), $(BUILD_SIMULATOR)$(file).d)

#Add all the defines in nvcc flags
NVCCFLAGS += $(DEFINES)

.PHONY:  all scheduler simulator .task .clustering .multicriteria .rabbit .json .json_s .libs
.PRECIOUS: $(BUILD_GPUSCHEDULER)%.o $(BUILD_SIMULATOR)%.o

all: scheduler simulator;

scheduler: .task .rabbit .clustering .multicriteria .libs .json $(LIST_GPUSCHEDULER)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $(BUILD_GPUSCHEDULER)*.o $(BUILD_GPUSCHEDULER)hierarchy/*.o $(NVOUT) $(GPUSCHEDULER_PATH)gpuscheduler.out

ifeq ($(MAKECMDGOALS),scheduler)
include $(LIST_RABBIT_DEP)
include $(LIST_CLUSTERING_DEP)
include $(LIST_MULTICRITERIA_DEP)
include $(LIST_GPUSCHEDULER_DEP)
endif

simulator:  .json_s $(LIST_SIMULATOR_OBJ)
	$(CXX) $(CXXFLAGS) $(BUILD_SIMULATOR)*.o -L$(RABBIT_LIBS_PATH) -lrabbitmq  $(COUT) $(SIMULATOR_PATH)simulator.out

ifeq ($(MAKECMDGOALS),simulator)
include $(LIST_SIMULATOR_DEP)
endif

.clustering: $(LIST_CLUSTERING)
ifeq ($(MAKECMDGOALS),clustering)
include $(LIST_CLUSTERING_DEP)
endif

.multicriteria: $(LIST_MULTICRITERIA)
ifeq ($(MAKECMDGOALS), multicriteria)
include $(LIST_MULTICRITERIA_DEP)
endif

.rabbit: $(LIST_RABBIT)
ifeq ($(MAKECMDGOALS), rabbit)
include $(LIST_RABBIT_DEP)
endif

.task: $(LIST_TASK)
ifeq ($(MAKECMDGOALS), task)
include $(LIST_TASK_DEP)
endif

$(BUILD_GPUSCHEDULER)json$(OBJ): $(GPUSCHEDULER_PATH)json.cpp
	$(CXX) $(CXXFLAGS_W/BOOST) $(CXX_OBJ) $< $(COUT)"$@";

$(BUILD_SIMULATOR)json$(OBJ): $(SIMULATOR_PATH)json.cpp
	$(CXX) $(CXXFLAGS) $(CXX_OBJ) $< $(COUT)"$@";


#Compiling all the objs in the final executable
$(BUILD_GPUSCHEDULER)%$(AUX) : $(BUILD_GPUSCHEDULER)%$(OBJ);

#Compile the multicriteria module
$(BUILD_GPUSCHEDULER)%$(OBJ) : $(MULTICRITERIA_PATH)%.cpp
	$(CXX) $(CXXFLAGS_W/BOOST) $(CXX_OBJ) $< $(COUT) $@;

$(BUILD_GPUSCHEDULER)%.d : $(MULTICRITERIA_PATH)%.cpp
	$(CXX) $(CXXFLAGS_W/BOOST) -M $< $(COUT) $@;

$(BUILD_GPUSCHEDULER)%$(OBJ) : $(MULTICRITERIA_PATH)%.cu
	$(NVCC) $(NVCCFLAGS) $(THIRDPARTY_FLAGS) $(NVCC_OBJ) $< $(NVOUT) $@;

$(BUILD_GPUSCHEDULER)%.d : $(MULTICRITERIA_PATH)%.cu
	$(NVCC) $(NVCCFLAGS) $(THIRDPARTY_FLAGS) -odir $(BUILD_GPUSCHEDULER) -M $< $(NVOUT) $@;
#Compile the rabbit module
$(BUILD_GPUSCHEDULER)%$(OBJ) : $(RABBIT_PATH)%.cpp
	$(CXX) $(CXXFLAGS_W/BOOST) $(CXX_OBJ) $< $(COUT) $@;

$(BUILD_GPUSCHEDULER)%.d : $(RABBIT_PATH)%.cpp
	$(CXX) $(CXXFLAGS_W/BOOST) -M $< $(COUT) $@;


#Compile the rabbit module
$(BUILD_GPUSCHEDULER)%$(OBJ) : $(TASKS_PATH)%.cpp
	$(CXX) $(CXXFLAGS_W/BOOST) $(CXX_OBJ) $< $(COUT) $@;

$(BUILD_GPUSCHEDULER)%.d : $(TASKS_PATH)%.cpp
	$(CXX) $(CXXFLAGS_W/BOOST) -M $< $(COUT) $@;

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

#Compile SIMULATOR module
$(BUILD_SIMULATOR)%$(OBJ) : $(SIMULATOR_PATH)%.cpp
	$(CXX) $(CXXFLAGS)  $(CXX_OBJ) $< $(COUT) $@;

$(BUILD_SIMULATOR)%.d : $(SIMULATOR_PATH)%.cpp
	$(CXX) $(CXXFLAGS) -M $< $(COUT) $@;

.libs: $(BUILD_GPUSCHEDULER)pugixml$(OBJ);

$(BUILD_GPUSCHEDULER)pugixml$(OBJ): $(VNE_PATH)libs/pugixml/src/pugixml.cpp
	$(CXX) $(CXXFLAGS_W/BOOST) $(CXX_OBJ) $< $(COUT)"$@";

.json: $(BUILD_GPUSCHEDULER)json$(OBJ);

.json_s: $(BUILD_SIMULATOR)json$(OBJ);

clear:
	rm -f $(BUILD_GPUSCHEDULER)*.o $(BUILD_GPUSCHEDULER)*.d;
	rm -f $(BUILD_SIMULATOR)*.o $(BUILD_SIMULATOR)*.d;
	rm -f $(GPUSCHEDULER_PATH)gpuscheduler.out
	rm -f $(SIMULATOR_PATH)simulator.out
	$(shell rm -r gpuScheduler/build/hierarchy)
	$(shell rm -r gpuScheduler/build)
