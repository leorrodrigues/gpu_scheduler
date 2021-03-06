#ifndef  _TASK_NOT_INCLUDED_
#define _TASK_NOT_INCLUDED_

#include <chrono>

#include "task_resources.hpp"
#include "pod.hpp"

class Task : public Task_Resources {
private:
//Time metrics variables
std::chrono::high_resolution_clock::time_point requested_time;
std::chrono::high_resolution_clock::time_point start_time;
std::chrono::high_resolution_clock::time_point stop_time;

//Pod variables
Pod** pods;
Container **containers;
unsigned int duration;
unsigned int early_duration;
unsigned int submission;

//Management variables
unsigned int pods_size;
unsigned int containers_size;
unsigned int links_size;
unsigned int allocated_time;
unsigned int delay;
unsigned int deadline;

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

void setRequestedTime();
void setStartTime();
void setStopTime();

void addDelay(unsigned int);
void setAllocatedTime(unsigned int);
void setSubmission(unsigned int);

void setLinkPath(int*);
void setLinkPathEdge(int*);
void setLinkDestination(int*);
void setLinkInit(int*);
void setLinkValues(float*);

std::chrono::high_resolution_clock::time_point getRequestedTime();
std::chrono::high_resolution_clock::time_point getStartTime();
std::chrono::high_resolution_clock::time_point getStopTime();

Pod** getPods();
Container** getContainers();
unsigned int getPodsSize();
unsigned int getDuration();
unsigned int getEarlyDuration();
unsigned int getSubmission();
unsigned int getContainersSize();
unsigned int getLinksSize();
unsigned int getAllocatedTime();
unsigned int getDelay();
unsigned int getDeadline();

int* getLinkPath();
int* getLinkPathEdge();
int* getLinkDestination();
int* getLinkInit();
float* getLinkValues();

float getBandwidthMax();
float getBandwidthMin();
float getBandwidthAllocated();

void updateBandwidth();

void print();

float taskUtility();
float linkUtility();

};

#endif
