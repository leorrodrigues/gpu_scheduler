#ifndef  _POD_NOT_INCLUDED_
#define _POD_NOT_INCLUDED_

#include "../../thirdparty/rapidjson/document.h"

#include "container.hpp"
#include "link.hpp"

class Pod {
private:
//Pod variables
Container** containers;
unsigned int id;
int host;

//Management variables
unsigned int containers_size;
unsigned int fit;

//Acumulative resources of the containers
float epc_min;
float ram_min;
float vcpu_min;
float epc_max;
float ram_max;
float vcpu_max;

public:

Pod();
Pod(unsigned int);
~Pod();

Container** getContainers();
unsigned int getContainersSize();

unsigned int getId();
int getHost();
unsigned int getFit();

float getEpcMin();
float getEpcMax();
float getRamMin();
float getRamMax();
float getVcpuMin();
float getVcpuMax();

void addContainer(Container*);
void setId(unsigned int);
void setHost(int);
void setFit(unsigned int);

friend std ::ostream& operator<<(std::ostream&, Pod const&);

};

#endif
