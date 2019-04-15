#ifndef _TYPE_NOT_DEFINED_
#define _TYPE_NOT_DEFINED_

#include <chrono>
#include <queue>
#include <vector>
#include <string>
#include <map>

#include "main_resources/main_resources_types.hpp"
#include "datacenter/tasks/task.hpp"

#include "thirdparty/spdlog/spdlog.h"
#include "thirdparty/spdlog/sinks/basic_file_sink.h"
#include "thirdparty/spdlog/sinks/stdout_color_sinks.h"

struct CompareTaskOnSubmission {
	bool operator()( Task* lhs, Task* rhs) const {
		if ((lhs->getSubmission()+lhs->getDelay()) == (rhs->getSubmission()+rhs->getDelay())) {
			return (lhs->getId()>rhs->getId());
		}else{
			return (lhs->getSubmission()+lhs->getDelay()) > (rhs->getSubmission()+rhs->getDelay());
		}
	}
};

struct CompareTaskOnDelete {
	bool operator()( Task* lhs, Task* rhs) const {
		return (lhs->getAllocatedTime()+lhs->getDuration()) > (rhs->getAllocatedTime()+rhs->getDuration());
	}
};

typedef struct {
	std::string multicriteria_method;
	std::string clustering_method;
	std::string topology_type;
	std::string standard;
	unsigned int data_type = 0;
	unsigned int topology_size=0;
	unsigned int current_time=0;
	unsigned int test_type=0;
	unsigned int request_size=0;
	unsigned int bw=0;
} options_t;

typedef struct {
	std::priority_queue<Task*, std::vector<Task*>, CompareTaskOnSubmission> tasks_to_allocate;
	std::priority_queue<Task*, std::vector<Task*>, CompareTaskOnDelete> tasks_to_delete;

	std::chrono::high_resolution_clock::time_point start;
	std::chrono::high_resolution_clock::time_point end;
} scheduler_t;

typedef struct total_resources_t : public main_resource_t {
	unsigned int links;
	unsigned int servers;
    float total_bandwidth=0;

	explicit total_resources_t() : main_resource_t(){
		links = 0;
		servers=0;
        total_bandwidth=0;
	}
} total_resources_t;

typedef struct consumed_resource_t : public main_resource_t {
	unsigned int time;
	unsigned int active_servers;
	unsigned int active_links;
    float total_bandwidth_consumed;

	explicit consumed_resource_t() : main_resource_t(){
		time=0;
		active_servers=0;
		active_links=0;
        total_bandwidth_consumed=0;
	}

} consumed_resource_t;

typedef struct {
	float dc_fragmentation=0;
	float link_fragmentation=0;
	float footprint=0;
	float vcpu_footprint = 0;
	float ram_footprint = 0;
	float link_footprint=0;
	unsigned int time=0;

} objective_function_t;

#endif
