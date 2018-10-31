#ifndef _UTILS_ALLOCATION_
#define _UTILS_ALLOCATION_

#include <iostream>

namespace Allocator {

inline bool checkFit(Host* host, Container* container){
	if(host->getResource()->mWeight["vcpu"]<container->containerResources->vcpu_max) {
		// std::cout<<"VCPU "<<host->getResource()->mWeight["vcpu"]<<" AND "<<container->containerResources->vcpu_max;
		return false;
	}
	if(host->getResource()->mWeight["memory"]<container->containerResources->ram_max) {
		// std::cout<<"Memory "<<host->getResource()->mWeight["memory"]<<" AND "<<container->containerResources->ram_max<<"\n";
		return false;
	}
	return true;
}

}
#endif
