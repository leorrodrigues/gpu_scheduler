#ifndef _TYPE_NOT_DEFINED_
#define _TYPE_NOT_DEFINED_

#include <vector>
#include <string>
#include <queue>
#include <map>

#include "datacenter/tasks/container.hpp"

namespace Allocation_t {
enum {
	NAIVE, DC, ALL, PURE, CLUSTERIZED
};
}

struct CompareContainerOnSubmission {
	bool operator()( Container* lhs, Container* rhs) const {
		if ((lhs->getSubmission()+lhs->getDelay()) == (rhs->getSubmission()+rhs->getDelay())) {
			return (lhs->getId()>rhs->getId());
		}else{
			return (lhs->getSubmission()+lhs->getDelay()) > (rhs->getSubmission()+rhs->getDelay());
		}
	}
};

struct CompareContainerOnDelete {
	bool operator()( Container* lhs, Container* rhs) const {
		return (lhs->getAllocatedTime()+lhs->getDuration()) > (rhs->getAllocatedTime()+rhs->getDuration());
	}
};

typedef struct {
	int allocation_type=Allocation_t::NAIVE;
	std::string multicriteria_method;
	std::string clustering_method;
	std::string topology_type;
	int topology_size=0;
	int current_time=0;
	int start_time=0;
	int test_type=0;
	int end_time=0;
	int request_size=0;
} options_t;

typedef struct {
	std::map<int, const char*> allocated_task;
	std::priority_queue<Container*, std::vector<Container*>, CompareContainerOnSubmission> containers_to_allocate;
	std::priority_queue<Container*, std::vector<Container*>, CompareContainerOnDelete> containers_to_delete;
	int total_containers=0;
	int total_accepted=0;
	int total_refused=0;
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
