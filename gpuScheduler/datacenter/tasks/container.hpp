#ifndef _CONTAINER_NOT_INCLUDED_
#define _CONTAINER_NOT_INCLUDED_

#include <iostream>

#include "task.hpp"

class Container : public Task {
private:
double duration;
double* links;
int id;
double submission;

typedef struct container_resources_t : public task_resource_t {
	int pod;
	double epc_min;
	double epc_max;
	double ram_min;
	double ram_max;
	double vcpu_max;
	double vcpu_min;
} container_resources_t;
protected:
public:

container_resources_t *containerResources;

Container();

void setTask(const char*);

container_resources_t* getResource();
double getDuration();
int getId();
double getSubmission();

};

#endif