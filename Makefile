####################
#        SCHEDULER      #
####################

$(shell mkdir -p scheduler/build)
$(shell mkdir -p scheduler/build/ahp)
$(shell mkdir -p scheduler/build/ahp/hierarchy)
$(shell mkdir -p scheduler/build/topsis)

#Programs
CXX = g++
NVCC = nvcc
LD = nvcc
RM = rm -f

#Folders
MULTICRITERIA_PATH = scheduler/multicriteria/
DATACENTER_PATH = scheduler/datacenter/
BUILD_GPUSCHEDULER = scheduler/build/
CLUSTERING_PATH = scheduler/clustering/
TASKS_PATH = scheduler/datacenter/tasks/
VNE_PATH = scheduler/thirdparty/vnegpu/
TOPOLOGY_PATH = scheduler/topology/
GPUSCHEDULER_PATH= scheduler/
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

CXXFLAGS = $(DEBUG_CXX) -std=c++17 -Wall -D_GLIBCXX_ASSERTIONS -D_FORTIFY_SOURCE=2 -fasynchronous-unwind-tables -fstack-protector-strong  -pipe -Werror=format-security  -Wduplicated-branches  -Wlogical-op  -Wnull-dereference  -Wdouble-promotion  -Wshadow  -Wformat=2 -Wduplicated-cond -fconcepts -O2

#CXXFLAGS = -std=c++17 -Wall -fconcepts

CXXFLAGS_W/BOOST = $(DEBUG_CXX) $(BOOSTFLAGS) -std=c++17 -Wall -D_GLIBCXX_ASSERTIONS -D_FORTIFY_SOURCE=2 -fasynchronous-unwind-tables -fstack-protector-strong  -pipe -Werror=format-security -fconcepts  -O2

#CXX_FLAGS_W/BOOST = $(BOOSTFLAGS) -std=c++17 -Wall -fconcepts

#nvcc
NVCCFLAGS = $(DEBUG_NVCC) -std=c++14 -Xptxas -O2 -use_fast_math -lineinfo

#NVCCFLAGS = -std=c++14 -Xptxas -lineinfo

#nve
GPUSCHEDULER_FLAG = -I "gpuScheduler"
THIRDPARTY_FLAGS = -I "scheduler/thirdparty/"

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

#Cluster module
CLUSTERING_FILES  := mclInterface

LIST_CLUSTERING := $(foreach file, $(CLUSTERING_FILES), $(BUILD_GPUSCHEDULER)$(file)$(AUX))

LIST_CLUSTERING_OBJ := $(foreach file, $(CLUSTERING_FILES_), $(BUILD_GPUSCHEDULER)$(file)$(OBJ))

LIST_CLUSTERING_DEP := $(foreach file, $(CLUSTERING_FILES), $(BUILD_GPUSCHEDULER)$(file).d)

#Multicriteria module
MULTICRITERIA_FILES := ahp/ahp ahp/ahpg ahp/hierarchy/hierarchy ahp/hierarchy/hierarchy_resource ahp/hierarchy/node ahp/hierarchy/edge topsis/topsis

LIST_MULTICRITERIA := $(foreach file,$(MULTICRITERIA_FILES), $(BUILD_GPUSCHEDULER)$(file)$(AUX))

LIST_MULTICRITERIA_OBJ := $(foreach file,$(MULTICRITERIA_FILES), $(BUILD_GPUSCHEDULER)$(file)$(OBJ))

LIST_MULTICRITERIA_DEP := $(foreach file, $(MULTICRITERIA_FILES), $(BUILD_GPUSCHEDULER)$(file).d)

#Tasks module
TASKS_FILES := container pod task

LIST_TASK := $(foreach file,$(TASKS_FILES), $(BUILD_GPUSCHEDULER)$(file)$(AUX))

LIST_TASK_OBJ := $(foreach file,$(TASKS_FILES), $(BUILD_GPUSCHEDULER)$(file)$(OBJ))

LIST_TASK_DEP := $(foreach file, $(TASKS_FILES), $(BUILD_GPUSCHEDULER)$(file).d)


#GPUSCHEDULER MODULE
GPUSCHEDULER_FILES := builder gpu_scheduler

LIST_GPUSCHEDULER := $(foreach file, $(GPUSCHEDULER_FILES), $(BUILD_GPUSCHEDULER)$(file)$(EXE))

LIST_GPUSCHEDULER_OBJ := $(foreach file, $(GPUSCHEDULER_FILES), $(BUILD_GPUSCHEDULER)$(file)$(OBJ))

LIST_GPUSCHEDULER_DEP := $(foreach file, $(GPUSCHEDULER_FILES), $(BUILD_GPUSCHEDULER)$(file).d)

#Add all the defines in nvcc flags
NVCCFLAGS += $(DEFINES)

.PHONY:  all scheduler .task .clustering .multicriteria  .json .json_s .libs
.PRECIOUS: $(BUILD_GPUSCHEDULER)%.o

all: scheduler;

scheduler: .task .clustering .multicriteria .libs .json $(LIST_GPUSCHEDULER)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $(BUILD_GPUSCHEDULER)*.o  $(BUILD_GPUSCHEDULER)ahp/*.o $(BUILD_GPUSCHEDULER)ahp/hierarchy/*.o $(BUILD_GPUSCHEDULER)topsis/*.o $(NVOUT) $(GPUSCHEDULER_PATH)gpuscheduler.out

ifeq ($(MAKECMDGOALS),scheduler)
include $(LIST_CLUSTERING_DEP)
include $(LIST_MULTICRITERIA_DEP)
include $(LIST_GPUSCHEDULER_DEP)
endif

.clustering: $(LIST_CLUSTERING)
ifeq ($(MAKECMDGOALS),clustering)
include $(LIST_CLUSTERING_DEP)
endif

.multicriteria: $(LIST_MULTICRITERIA)
ifeq ($(MAKECMDGOALS), multicriteria)
include $(LIST_MULTICRITERIA_DEP)
endif

.task: $(LIST_TASK)
ifeq ($(MAKECMDGOALS), task)
include $(LIST_TASK_DEP)
endif

$(BUILD_GPUSCHEDULER)json$(OBJ): $(GPUSCHEDULER_PATH)json.cpp
	$(CXX) $(CXXFLAGS_W/BOOST) $(CXX_OBJ) $< $(COUT)"$@";

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

#Compile the tasks module
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

.libs: $(BUILD_GPUSCHEDULER)pugixml$(OBJ);

$(BUILD_GPUSCHEDULER)pugixml$(OBJ): $(VNE_PATH)libs/pugixml/src/pugixml.cpp
	$(CXX) $(CXXFLAGS_W/BOOST) $(CXX_OBJ) $< $(COUT)"$@";

.json: $(BUILD_GPUSCHEDULER)json$(OBJ);

clear:
	rm -f $(BUILD_GPUSCHEDULER)*.o $(BUILD_GPUSCHEDULER)*.d;
	rm -f $(GPUSCHEDULER_PATH)gpuscheduler.out
	# $(shell rm -r scheduler/build/hierarchy)
	$(shell rm -r scheduler/build)
