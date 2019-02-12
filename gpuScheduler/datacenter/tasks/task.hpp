#ifndef  _TASK_NOT_INCLUDED_
#define _TASK_NOT_INCLUDED_

#include "../../thirdparty/rapidjson/document.h"

#include "task_resources.hpp"
#include "pod.hpp"

class Task : public Task_Resources {
private:
//Pod variables
Pod** pods;
unsigned int duration;
unsigned int submission;

//Management variables
unsigned int pods_size;
unsigned int containers_size;
unsigned int links_size;
unsigned int allocated_time;
unsigned int delay;
unsigned int fit;

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
unsigned int getSubmission();
unsigned int getContainersSize();
unsigned int getAllocatedTime();
unsigned int getDelay();
unsigned int getFit();

friend std ::ostream& operator<<(std::ostream&, Task const&);

};

#endif
