#ifndef  _POD_NOT_INCLUDED_
#define _POD_NOT_INCLUDED_

#include "../../thirdparty/rapidjson/document.h"

#include "container.hpp"

class Pod : public Task {
private:
Container** containers;
unsigned int containers_size;

double* links;
unsigned int links_size;

unsigned int duration;
unsigned int id;
unsigned int submission;
unsigned int allocated_time;
unsigned int delay;
unsigned int fit;

typedef struct pod_resources_t : public task_resource_t {
	double epc_min;
	double ram_min;
	double vcpu_min;
	double epc_max;
	double ram_max;
	double vcpu_max;
} pod_resources_t;

public:

pod_resources_t* podResources;

Pod();
~Pod();

void setSubmission(unsigned int);
void setAllocatedTime(unsigned int);
void setFit(unsigned int);
void addDelay();
void addDelay(unsigned int);

pod_resources_t* getResource();
unsigned int getDuration();
unsigned int getId();
unsigned int getAllocatedTime();
unsigned int getFit();
unsigned int getDelay();

void addContainer(Container*);



};

#endif
