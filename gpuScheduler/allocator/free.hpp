#ifndef _FREE_ALLOCATION_
#define _FREE_ALLOCATION_

#include <iostream>

#include "../datacenter/tasks/container.hpp"
#include "../datacenter/host.hpp"

namespace Allocator {
bool freeHostResource(Host* host, Container* container, consumed_resource_t* consumed){
	if(host==NULL) {
		std::cerr << " Free Host Resource received NULL host\n";
		exit(2);
	}

	host->removeContainer(container);
	host->removeAllocaredResource();

	int fit = container->getFit();
	if(fit==7) { // allocate MAX VCPU AND RAM
		consumed->ram -= container->containerResources->ram_max;
		consumed->vcpu -=container->containerResources->vcpu_max;
	}else if(fit==8) { // ALLOCATE MAX VCPU AND RAM MIN
		consumed->ram -= container->containerResources->ram_min;
		consumed->vcpu -= container->containerResources->vcpu_max;
	}else if(fit==10) { // ALLOCATE VCPU MIN AND RAM MAX
		consumed->ram -= container->containerResources->ram_max;
		consumed->vcpu -=container->containerResources->vcpu_min;
	}else if(fit==11) { // ALLOCATE VCPU AND RAM MIN
		consumed->ram -= container->containerResources->ram_min;
		consumed->vcpu -= container->containerResources->vcpu_min;
	}

	return true;
}
}
#endif
