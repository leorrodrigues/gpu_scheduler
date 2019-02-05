#ifndef _CONTAINER_NOT_INCLUDED_
#define _CONTAINER_NOT_INCLUDED_

#include <iostream>

#include "task.hpp"

class Container : public Task {
private:

typedef struct container_resources_t : public task_resource_t {
	unsigned int pod;
	unsigned int name;
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
~Container();

void setTask(const char*);
void setSubmission(int);
void setAllocatedTime(int);
void setFit(int);
void addDelay();
void addDelay(int);

container_resources_t* getResource();
int getDuration();
int getId();
int getSubmission();
int getAllocatedTime();
int getFit();
int getDelay();

friend std ::ostream& operator<<(std::ostream&, Container const&);

};

#endif
