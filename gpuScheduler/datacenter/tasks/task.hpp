#ifndef  _TASK_NOT_INCLUDED_
#define _TASK_NOT_INCLUDED_

#include "task_resources.hpp"
#include "pod.hpp"

class Task : public Task_Resources {
private:
//Pod variables
Pod** pods;
Container **containers;
unsigned int duration;
unsigned int submission;

//Management variables
unsigned int pods_size;
unsigned int containers_size;
unsigned int links_size;
unsigned int allocated_time;
unsigned int delay;

public:

Task();
~Task();

void setTask(const char*);

void addDelay();
void addDelay(unsigned int);
void setAllocatedTime(unsigned int);
void setSubmission(unsigned int);

Pod** getPods();
Container** getContainers();
unsigned int getPodsSize();
unsigned int getDuration();
unsigned int getSubmission();
unsigned int getContainersSize();
unsigned int getLinksSize();
unsigned int getAllocatedTime();
unsigned int getDelay();

friend std ::ostream& operator<<(std::ostream&, Task const&);

};

#endif
