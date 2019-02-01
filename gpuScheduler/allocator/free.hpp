#ifndef _FREE_ALLOCATION_
#define _FREE_ALLOCATION_

#include <iostream>

#include "../datacenter/tasks/container.hpp"
#include "../datacenter/host.hpp"

namespace Allocator {
bool freeHostResource(Host* host, Container* container, consumed_resource_t* consumed, Builder* builder){
	if(host==NULL) {
		std::cerr << " Free Host Resource received NULL host\n";
		exit(2);
	}

	//Update the host resources
	host->removeContainer(container);
	host->removeAllocaredResource();

	//Need to find the group that has this host and update their resource
	std::vector<Host*> groups = builder->getClusterHosts();
	// int index=-1;
	for(int i=0; i<groups.size(); i++) {
		if(builder->findHostInGroup(groups[i]->getId(),host->getId())) {
			// index=i;
			groups[i]->removeContainer(container);
			break;
		}
	}
	// if(index==-1) {
	//      printf("free.hpp(30) Erro host isn't in any group!\n");
	//      exit(0);
	// }

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
