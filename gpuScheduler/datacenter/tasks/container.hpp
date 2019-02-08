#ifndef _CONTAINER_NOT_INCLUDED_
#define _CONTAINER_NOT_INCLUDED_

#include <iostream>

#include "link.hpp"

class Container {
private:

Link **links;
unsigned int links_size;

int host;
unsigned int name;
float epc_min;
float epc_max;
float ram_min;
float ram_max;
float vcpu_max;
float vcpu_min;


protected:
public:

Container();
~Container();

unsigned int getHost();
unsigned int getName();
float getEpcMin();
float getEpcMax();
float getRamMin();
float getRamMax();
float getVcpuMax();
float getVcpuMin();

void setPod(unsigned int);
void setName(unsigned int);
void setEpcMin(float);
void setEpcMax(float);
void setRamMin(float);
void setRamMax(float);
void setVcpuMax(float);
void setVcpuMin(float);
void setLink(Link*);

friend std ::ostream& operator<<(std::ostream&, Container const&);

};

#endif
