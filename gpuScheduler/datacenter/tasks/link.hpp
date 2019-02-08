#ifndef  _LINK_NOT_INCLUDED_
#define _LINK_NOT_INCLUDED_

#include "../../thirdparty/rapidjson/document.h"

#include <string>

class Link {
private:

unsigned int souce;
unsigned int destination;
std::string name;
float bandwidth_max;
float bandwidth_min;

public:

Link();
~Link();

void setSource(unsigned int);
void setDestination(unsigned int);
void setName(const char*);
void setBandwidthMax(float);
void setBandwidthMin(float);

unsigned int getSource();
unsigned int getDestination();
const char* getName();
float getBandwidthMax();
float getBandwidthMin();

friend std ::ostream& operator<<(std::ostream&, Link const&);

};

#endif
