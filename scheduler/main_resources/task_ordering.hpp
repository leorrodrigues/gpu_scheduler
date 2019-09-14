#ifndef _TASK_ORDERING_NOT_DEFINED_
#define _TASK_ORDERING_NOT_DEFINED_

#include <vector>
#include <string>
#include <queue>
#include <map>

#include "main_resources_types.hpp"
#include "../datacenter/tasks/task.hpp"

struct TaskOnDelete_CMP {
	bool operator()( Task* lhs, Task* rhs) const {
		return (lhs->getAllocatedTime()+lhs->getDuration()) > (rhs->getAllocatedTime()+rhs->getDuration());
	}
};

class AbstractPQ {
public:
virtual void push(Task*) = 0;
virtual void pop() = 0;
virtual Task* top() = 0;
virtual bool empty() = 0;
virtual int size() = 0;
};

class FCFS : public AbstractPQ {
private:
struct FCFS_CMP {         //  first-come-first-served
	bool operator()( Task* lhs, Task* rhs) const {
		return (lhs->getSubmission() + lhs->getDelay()) > (rhs->getSubmission() + rhs->getDelay());
	}
};
std::priority_queue<Task*, std::vector<Task*>, FCFS_CMP> queue;
public:
void push(Task* t){
	queue.push(t);
}

void pop(){
	queue.pop();
}

Task* top(){
	return queue.top();
}

bool empty(){
	return queue.empty();
}

int size(){
	return queue.size();
}
};

class SPF : public AbstractPQ {
private:
struct SPF_CMP {         // smallest estimated processing time first
	bool operator()( Task* lhs, Task* rhs) const {
		auto lhs_time = lhs->getSubmission() + lhs->getDelay();
		auto rhs_time = rhs->getSubmission() + rhs->getDelay();
		if (lhs_time == rhs_time) {
			return ((lhs->getDuration() - lhs->getSubmission()) > (rhs->getDuration() - lhs->getSubmission()));
		}else{
			return (lhs_time > rhs_time);
		}
	}
};
std::priority_queue<Task*, std::vector<Task*>, SPF_CMP> queue;
public:
void push(Task* t){
	queue.push(t);
}

void pop(){
	queue.pop();
}

Task* top(){
	return queue.top();
}

bool empty(){
	return queue.empty();
}

int size(){
	return queue.size();
}
};

class SQFMIN : public AbstractPQ {
private:
struct SQFMIN_CMP { // smallest min resource requirement first
	bool operator()( Task* lhs, Task* rhs) const {
		if ((lhs->getSubmission()+lhs->getDelay()) == (rhs->getSubmission()+rhs->getDelay())) {
			return (lhs->getAllMinResource() > rhs->getAllMinResource());
		}else{
			return (lhs->getSubmission()+lhs->getDelay() > rhs->getSubmission()+rhs->getDelay());
		}
	}
};
std::priority_queue<Task*, std::vector<Task*>, SQFMIN_CMP> queue;
public:
void push(Task* t){
	queue.push(t);
}

void pop(){
	queue.pop();
}

Task* top(){
	return queue.top();
}

bool empty(){
	return queue.empty();
}

int size(){
	return queue.size();
}
};

class SQFMAX : public AbstractPQ {
private:
struct SQFMAX_CMP { // smallest max resource requirement first
	bool operator()( Task* lhs, Task* rhs) const {
		if ((lhs->getSubmission()+lhs->getDelay()) == (rhs->getSubmission()+rhs->getDelay())) {
			return (lhs->getAllMaxResource() > rhs->getAllMaxResource());
		}else{
			return (lhs->getSubmission()+lhs->getDelay() > rhs->getSubmission()+rhs->getDelay());
		}
	}
};
std::priority_queue<Task*, std::vector<Task*>, SQFMAX_CMP> queue;
public:
void push(Task* t){
	queue.push(t);
}

void pop(){
	queue.pop();
}

Task* top(){
	return queue.top();
}

bool empty(){
	return queue.empty();
}

int size(){
	return queue.size();
}
};

class SAFMIN : public AbstractPQ {
private:
struct SAFMIN_CMP { // smallest min estimated area first
	bool operator()( Task* lhs, Task* rhs) const {
		if ((lhs->getSubmission()+lhs->getDelay()) == (rhs->getSubmission()+rhs->getDelay())) {
			float lhs_pt = lhs->getDuration() - lhs->getSubmission();
			float rhs_pt = rhs->getDuration() - rhs->getSubmission();
			return ((lhs_pt * lhs->getAllMinResource()) > (rhs_pt * rhs->getAllMinResource()));
		}else{
			return (lhs->getSubmission()+lhs->getDelay() > rhs->getSubmission()+rhs->getDelay());
		}
	}
};
std::priority_queue<Task*, std::vector<Task*>, SAFMIN_CMP> queue;
public:
void push(Task* t){
	queue.push(t);
}

void pop(){
	queue.pop();
}

Task* top(){
	return queue.top();
}

bool empty(){
	return queue.empty();
}

int size(){
	return queue.size();
}
};

class SAFMAX : public AbstractPQ {
private:
struct SAFMAX_CMP { // smallest max estimated area first
	bool operator()( Task* lhs, Task* rhs) const {
		if ((lhs->getSubmission()+lhs->getDelay()) == (rhs->getSubmission()+rhs->getDelay())) {
			float lhs_pt = lhs->getDuration() - lhs->getSubmission();
			float rhs_pt = rhs->getDuration() - rhs->getSubmission();
			return ((lhs_pt * lhs->getAllMaxResource()) > (rhs_pt * rhs->getAllMaxResource()));
		}else{
			return (lhs->getSubmission()+lhs->getDelay() > rhs->getSubmission()+rhs->getDelay());
		}
	}
};
std::priority_queue<Task*, std::vector<Task*>, SAFMAX_CMP> queue;
public:
void push(Task* t){
	queue.push(t);
}

void pop(){
	queue.pop();
}

Task* top(){
	return queue.top();
}

bool empty(){
	return queue.empty();
}

int size(){
	return queue.size();
}
};

class SDAFMIN : public AbstractPQ {
private:
struct SDAFMIN_CMP { // smallest difference estimated area first
	bool operator()( Task* lhs, Task* rhs) const {
		if ((lhs->getSubmission()+lhs->getDelay()) == (rhs->getSubmission()+rhs->getDelay())) {
			float lhs_pt = lhs->getDuration() - lhs->getSubmission();
			float rhs_pt = rhs->getDuration() - rhs->getSubmission();
			return ((lhs_pt * lhs->getAllDiffResource()) > (rhs_pt * rhs->getAllDiffResource()));
		}else{
			return (lhs->getSubmission()+lhs->getDelay() > rhs->getSubmission()+rhs->getDelay());
		}
	}
};
std::priority_queue<Task*, std::vector<Task*>, SDAFMIN_CMP> queue;
public:
void push(Task* t){
	queue.push(t);
}

void pop(){
	queue.pop();
}

Task* top(){
	return queue.top();
}

bool empty(){
	return queue.empty();
}

int size(){
	return queue.size();
}
};

#endif
