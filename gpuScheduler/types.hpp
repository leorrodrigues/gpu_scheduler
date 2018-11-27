#ifndef _TYPE_NOT_DEFINED_
#define _TYPE_NOT_DEFINED_

#include <vector>
#include <string>
#include <map>

#include "datacenter/tasks/container.hpp"

namespace Allocation_t {
enum {
	NAIVE, DC, ALL
};
}

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
} options_t;

typedef struct {
	std::map<int, const char*> allocated_task;
	std::vector<Container*> containers;
	int total_containers=0;
	int total_accepted=0;
	int total_refused=0;
} scheduler_t;

typedef struct {
	float vcpu=0;
	float ram=0;
	int servers=0;
} total_resources_t;

typedef struct {
	int time=0;
	float vcpu=0;
	float ram=0;
	int active_servers=0;
} consumed_resource_t;

typedef struct {
	int time = 0;
	float fragmentation = 0;
	float footprint = 0;
	float vcpu_footprint = 0;
	float ram_footprint = 0;
} objective_function_t;

#endif
