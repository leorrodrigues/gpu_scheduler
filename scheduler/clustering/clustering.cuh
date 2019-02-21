#ifndef _CUSTERING_NOT_INCLUDED_
#define _CUSTERING_NOT_INCLUDED_

#include <vector>
#include <string>
#include <map>
#include <set>

#include "../datacenter/host.hpp"

#include "../topology/fatTreeInterface.cuh"
#include "../topology/bcubeInterface.cuh"
#include "../topology/dcellInterface.cuh"

class Clustering {
public:

virtual ~Clustering()=0;

virtual void run(Topology*)=0;

//virtual std::map<int,Host*> getResult(Topology* topology,std::vector<Host*>)=0;
virtual std::vector<Host*> getResult(Topology* topology,std::vector<Host*>)=0;

virtual int getHostsMedianInGroup()=0;
virtual std::vector<Host*> getHostsInGroup(int)=0;
virtual bool findHostInGroup(unsigned int, unsigned int)=0;

virtual void listGroups(Topology*)=0;

virtual int getResultSize()=0;

};

inline Clustering::~Clustering(){
}
#endif
