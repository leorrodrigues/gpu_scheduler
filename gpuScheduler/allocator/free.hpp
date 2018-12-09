#ifndef _FREE_ALLOCATION_
#define _FREE_ALLOCATION_

#include <iostream>

#include "../datacenter/tasks/container.hpp"
#include "../datacenter/host.hpp"

namespace Allocator {
bool freeHostResource(Host* host, Container* container){
	if(host==NULL) {
		std::cerr << " Free Host Resource received NULL host\n";
		exit(2);
	}
	// std::cout << "Free Host Resource Function\n";
	host->removeContainer(container);
	host->removeAllocaredResource();
	return true;
}
}
#endif
