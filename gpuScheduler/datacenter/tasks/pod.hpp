#ifndef  _POD_NOT_INCLUDED_
#define _POD_NOT_INCLUDED_

#include "../../thirdparty/rapidjson/document.h"

#include "container.hpp"

class Pod {
private:
//Pod variables
Container** containers;
float* links;
unsigned int duration;
unsigned int id;
unsigned int submission;

//Management variables
unsigned int containers_size;
unsigned int links_size;
unsigned int allocated_time;
unsigned int delay;
unsigned int fit;

//Acumulative resources of the containers
float epc_min;
float ram_min;
float vcpu_min;
float epc_max;
float ram_max;
float vcpu_max;

public:

Pod();
~Pod();

void setTask(const char*);

void addDelay();
void addDelay(unsigned int);
void setFit(unsigned int);
void setAllocatedTime(unsigned int);
void setSubmission(unsigned int);

Container** getContainers();
float* getLinks();
unsigned int getDuration();
unsigned int getId();
unsigned int getSubmission();
unsigned int getContainersSize();
unsigned int getLinksSize();
unsigned int getAllocatedTime();
unsigned int getDelay();
unsigned int getFit();
float getEpcMin();
float getEpcMax();
float getRamMin();
float getRamMax();
float getVcpuMin();
float getVcpuMax();

void addContainer(Container*);

friend std ::ostream& operator<<(std::ostream&, Pod const&);

};

#endif
