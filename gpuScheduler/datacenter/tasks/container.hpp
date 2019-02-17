#ifndef _CONTAINER_NOT_INCLUDED_
#define _CONTAINER_NOT_INCLUDED_

#include "task_resources.hpp"

typedef struct {
	unsigned int destination;
	float bandwidth_max;
	float bandwidth_min;
} Link;

class Container : public Task_Resources {
private:
Link *links;
unsigned int links_size;
unsigned int host_id;

public:

Container();
~Container();

Link* getLinks();

unsigned int getLinksSize();

unsigned int getHostId();

void setLink(unsigned int, float, float);

void setHostId(unsigned int);

friend std ::ostream& operator<<(std::ostream&, Container const&);

};

#endif
