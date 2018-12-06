#ifndef _UTILS_ALLOCATION_
#define _UTILS_ALLOCATION_

#include <iostream>

namespace Allocator {

inline bool checkFit(Host* host, Container* container){
	if(host->getResource()->mFloat["vcpu"]<container->containerResources->vcpu_max) {
		// std::cout<<"VCPU "<<host->getResource()->mFloat["vcpu"]<<" AND "<<container->containerResources->vcpu_max;
		return false;
	}
	if(host->getResource()->mFloat["memory"]<container->containerResources->ram_max) {
		// std::cout<<"Memory "<<host->getResource()->mFloat["memory"]<<" AND "<<container->containerResources->ram_max<<"\n";
		return false;
	}
	return true;
}

}
#endif
