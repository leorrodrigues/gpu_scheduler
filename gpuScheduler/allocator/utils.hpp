#ifndef _UTILS_ALLOCATION_
#define _UTILS_ALLOCATION_

#include <iostream>

namespace Allocator {

inline int checkFit(Host* host, Container* container){
	// 7 VCPU AND RAM MAX
	// 8 VCPU MAX RAM MIN
	// 10 VCPU MIN RAM MAX
	// 11 VCPU MIN RAM MIN
	// 0 NOT FIT
	int total=0;
	if(host->getResource()["vcpu"]>=container->containerResources->vcpu_max) {
		// std::cout<<"VCPU "<<host->getResource()->mFloat["vcpu"]<<" AND "<<container->containerResources->vcpu_max;
		total+=1;
	}else if(host->getResource()["vcpu"]>=container->containerResources->vcpu_min) {
		total+=4;
	}else{
		total=0;
	}
	if(host->getResource()["memory"]>=container->containerResources->ram_max) {
		// std::cout<<"Memory "<<host->getResource()->mFloat["memory"]<<" AND "<<container->containerResources->ram_max<<"\n";
		total+=6;
	} else if(host->getResource()["memory"]>=container->containerResources->ram_min) {
		total+=7;
	}else{
		total=0;
	}
	return total;
}

}
#endif
