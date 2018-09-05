#ifndef _TASK_NOT_INCLUDED_
#define _TASK_NOT_INCLUDED_

#include <cstddef>

class Task {
protected:
typedef struct {
	int name;
} task_resource_t;
private:
public:

virtual void setTask(const char*)=0;

virtual task_resource_t* getResource()=0;

virtual double getDuration()=0;
virtual int getId()=0;
virtual double getSubmission()=0;

};

#endif
