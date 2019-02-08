#ifndef _FREE_ALLOCATION_
#define _FREE_ALLOCATION_

#include <iostream>

#include "../datacenter/tasks/pod.hpp"
#include "../datacenter/host.hpp"

namespace Allocator {
bool freeHostResource(Host* host, Task* task, consumed_resource_t* consumed, Builder* builder){
	if(host==NULL) {
		std::cerr << " Free Host Resource received NULL host\n";
		exit(2);
	}

	//Need to iterate through ALL the pods and get the hosts that they are allocated.

	//Update the host resources
	host->removePod(pod);
	host->removeAllocaredResource();

	//Need to find the group that has this host and update their resource
	std::vector<Host*> groups = builder->getClusterHosts();
	// int index=-1;
	for(int i=0; i<groups.size(); i++) {
		if(builder->findHostInGroup(groups[i]->getId(),host->getId())) {
			// index=i;
			groups[i]->removePod(pod);
			break;
		}
	}
	// if(index==-1) {
	//      printf("free.hpp(30) Erro host isn't in any group!\n");
	//      exit(0);
	// }

	int fit = pod->getFit();
	// The pod was allocated, so the consumed variable has to be updated
	if(fit==7) { // allocate MAX VCPU AND RAM
		consumed->ram  -= pod->getRamMax();
		consumed->vcpu -= pod->getVcpuMax();
	}else if(fit==8) { // ALLOCATE MAX VCPU AND RAM MIN
		consumed->ram  -= pod->getRamMin();
		consumed->vcpu -= pod->getVcpuMax();
	}else if(fit==10) { // ALLOCATE VCPU MIN AND RAM MAX
		consumed->ram  -= pod->getRamMax();
		consumed->vcpu -= pod->getVcpuMin();
	}else if(fit==11) { // ALLOCATE VCPU AND RAM MIN
		consumed->ram  -= pod->getRamMin();
		consumed->vcpu -= pod->getVcpuMin();
	}
	return true;
}
}
#endif
