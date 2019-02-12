#ifndef _TYPE_NOT_DEFINED_
#define _TYPE_NOT_DEFINED_

#include <vector>
#include <string>
#include <queue>
#include <map>

#include "datacenter/tasks/task.hpp"

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
	unsigned int topology_size=0;
	unsigned int current_time=0;
	unsigned int test_type=0;
	unsigned int request_size=0;
	unsigned int standard=0;
} options_t;

typedef struct {
	std::priority_queue<Task*, std::vector<Task*>, CompareTaskOnSubmission> tasks_to_allocate;
	std::priority_queue<Task*, std::vector<Task*>, CompareTaskOnDelete> tasks_to_delete;
} scheduler_t;

typedef struct {
	double vcpu=0;
	double ram=0;
	int servers=0;
} total_resources_t;

typedef struct {
	int time=0;
	double vcpu=0;
	double ram=0;
	int active_servers=0;
} consumed_resource_t;

typedef struct {
	int time = 0;
	double fragmentation = 0;
	double footprint = 0;
	double vcpu_footprint = 0;
	double ram_footprint = 0;
} objective_function_t;

#endif
