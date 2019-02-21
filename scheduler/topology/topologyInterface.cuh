#ifndef _TOPOLOGY_NOT_INCLUDED_
#define _TOPOLOGY_NOT_INCLUDED_

#include <vnegpu/graph.cuh>
#include <iostream>
#include <iomanip>

class Topology {
public:

virtual ~Topology()=0;

virtual void setTopology()=0;

virtual void setSize(int)=0;
virtual void setLevel(int)=0;

virtual void setResource(std::map<std::string, float> resource)=0;

virtual void populateTopology(std::vector<Host*>)=0;

virtual vnegpu::graph<float>* getGraph()=0;

virtual std::string getTopology()=0;

virtual int getIndexEdge()=0;

virtual void listTopology()=0;
};

inline Topology::~Topology(){
}
#endif
