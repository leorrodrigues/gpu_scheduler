#ifndef _TOPOLOGY_NOT_INCLUDED_
#define _TOPOLOGY_NOT_INCLUDED_

#include "../thirdparty/spdlog/spdlog.h"
#include "../thirdparty/spdlog/sinks/stdout_color_sinks.h"

#include "../main_resources/interval_tree.hpp"

#include <vnegpu/graph.cuh>
#include <iostream>
#include <iomanip>

class Topology {
public:

virtual ~Topology() = 0;

virtual void setTopology() = 0;

virtual void setSize(int) = 0;
virtual void setLevel(int) = 0;

virtual void setResource(std::map<std::string, Interval_Tree::Interval_Tree*> resource) = 0;

virtual void populateTopology(std::vector<Host*>) = 0;

virtual vnegpu::graph<float>* getGraph() = 0;

virtual std::string getTopology() = 0;

virtual void listTopology() = 0;
};

inline Topology::~Topology(){
}
#endif
