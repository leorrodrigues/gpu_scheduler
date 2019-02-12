#ifndef _CONTAINER_NOT_INCLUDED_
#define _CONTAINER_NOT_INCLUDED_

#include <iostream>

#include "task_resources.hpp"
#include "link.hpp"

class Container : public Task_Resources {
private:

Link *links;
unsigned int links_size;

public:

Container();
~Container();

Link* getLinks();

void setLink(Link);
void setLink(unsigned int, float, float);

friend std ::ostream& operator<<(std::ostream&, Container const&);

};

#endif
