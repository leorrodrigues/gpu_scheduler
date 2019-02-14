#ifndef  _POD_NOT_INCLUDED_
#define  _POD_NOT_INCLUDED_

#include "../../thirdparty/rapidjson/document.h"

#include "../host.hpp"
#include "container.hpp"

class Pod : public Task_Resources {
private:
//Pod variables
Container** containers;
Host* host;

//Management variables
unsigned int containers_size;
public:

Pod();
Pod(unsigned int);
~Pod();

Container** getContainers();
unsigned int getContainersSize();

Host* getHost();

void addContainer(Container*);
void setHost(Host*);

friend std ::ostream& operator<<(std::ostream&, Pod const&);

};

#endif
