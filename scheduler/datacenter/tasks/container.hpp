#ifndef _CONTAINER_NOT_INCLUDED_
#define _CONTAINER_NOT_INCLUDED_

#include "task_resources.hpp"

typedef struct {
	float bandwidth_max;
	float bandwidth_min;
	unsigned int destination;
} Link;

class Container : public Task_Resources {
private:
Link *links;

float total_bandwidth_max;
float total_bandwidth_min;

unsigned int links_size;
unsigned int host_id; //represents the id of the host in the scheduler vector
unsigned int host_idg; //represents the id of the host in the vne::graph

public:

Container();
~Container();

Link* getLinks();

float getBandwidthMax();
float getBandwidthMin();

unsigned int getLinksSize();

unsigned int getHostId();
unsigned int getHostIdg();

void setLink(unsigned int, float, float);

void setHostId(unsigned int);
void setHostIdg(unsigned int);

void print();

};

#endif
