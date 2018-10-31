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

std::map<int,std::vector<Host*> > host_groups;

public:
MCLInterface();

void run(Topology*);

//std::map<int,Host*> getResult(Topology* topology,std::vector<Host*>);
std::vector<Host*> getResult(Topology*,std::vector<Host*>);

int getHostsMedianInGroup();
std::vector<Host*> getHostsInGroup(int);

void listGroups(Topology*);

};
#endif
