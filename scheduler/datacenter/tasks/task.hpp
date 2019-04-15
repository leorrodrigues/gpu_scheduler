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
unsigned int delay_dc;
unsigned int delay_link;

//Link management variables
// This variables are used during the allocation and desalocation of links
// Their construction are made in links_allocator function
float * values;
int *path;
int *path_edge;
int *destination;
int *init;

public:

Task();
~Task();

void setTask(const char*);

void addDelay();
void addDelay(unsigned int);
void addDelayDC(unsigned int);
void addDelayLink(unsigned int);
void setAllocatedTime(unsigned int);
void setSubmission(unsigned int);

void setLinkPath(int*);
void setLinkPathEdge(int*);
void setLinkDestination(int*);
void setLinkInit(int*);
void setLinkValues(float*);

Pod** getPods();
Container** getContainers();
unsigned int getPodsSize();
unsigned int getDuration();
unsigned int getSubmission();
unsigned int getContainersSize();
unsigned int getLinksSize();
unsigned int getAllocatedTime();
unsigned int getDelay();
unsigned int getDelayDC();
unsigned int getDelayLink();

int* getLinkPath();
int* getLinkPathEdge();
int* getLinkDestination();
int* getLinkInit();
float* getLinkValues();

float getBandwidthMax();
float getBandwidthMin();

void updateBandwidth();

void print();

float taskUtility();
float linkUtility();

};

#endif
