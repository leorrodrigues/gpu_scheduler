#ifndef _MCLINTERFACE_NOT_INCLUDED_
#define _MCLINTERFACE_NOT_INCLUDED_

#include <vector>
#include <string>
#include <map>

#include "clustering.cuh"

#include <vnegpu/algorithm/mcl.cuh>

class MCLInterface : public Clustering {
private:
vnegpu::graph<float>* dataCenter;

public:
MCLInterface();

void run(Topology*);

//std::map<int,Host*> getResult(Topology* topology,std::vector<Host*>);
std::vector<Host*> getResult(Topology* topology,std::vector<Host*>);

void listGroups(Topology*);

};
#endif
