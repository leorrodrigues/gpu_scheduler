#ifndef  _TASK_NOT_INCLUDED_
#define _TASK_NOT_INCLUDED_

#include "../../thirdparty/rapidjson/document.h"

#include "container.hpp"
#include "pod.hpp"

class Task {
private:
//Pod variables
Pod** pods;
unsigned int duration;
unsigned int id;
unsigned int submission;

//Management variables
unsigned int pods_size;
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

Task();
~Task();

void setTask(const char*);

void addDelay();
void addDelay(unsigned int);
void setFit(unsigned int);
void setAllocatedTime(unsigned int);
void setSubmission(unsigned int);

Pod** getPods();
unsigned int getPodsSize();
unsigned int getDuration();
unsigned int getId();
unsigned int getSubmission();
unsigned int getContainersSize();
unsigned int getAllocatedTime();
unsigned int getDelay();
unsigned int getFit();
float getEpcMin();
float getEpcMax();
float getRamMin();
float getRamMax();
float getVcpuMin();
float getVcpuMax();

friend std ::ostream& operator<<(std::ostream&, Task const&);

};

#endif
