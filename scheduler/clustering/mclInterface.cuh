
#ifndef _MCLINTERFACE_NOT_INCLUDED_
#define _MCLINTERFACE_NOT_INCLUDED_

#include "clustering.cuh"

#include "../thirdparty/vnegpu/algorithm/mcl.cuh"

class MCLInterface : public Clustering {
private:
vnegpu::graph<float>* dataCenter;
int resultSize;
std::map<int,std::vector<Host*> > host_groups;

public:
MCLInterface();
~MCLInterface();

void run(Topology*);

//std::map<int,Host*> getResult(Topology* topology,std::vector<Host*>);
std::vector<Host*> getResult(Topology*,std::vector<Host*>);
int getResultSize();
int getHostsMedianInGroup();
std::vector<Host*> getHostsInGroup(int);
bool findHostInGroup(unsigned int, unsigned int);

void listGroups(Topology*);

};
#endif
