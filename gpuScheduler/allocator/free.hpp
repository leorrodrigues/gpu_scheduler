#ifndef _FREE_ALLOCATION_
#define _FREE_ALLOCATION_

#include <iostream>

#include "../datacenter/tasks/pod.hpp"
#include "../datacenter/host.hpp"

namespace Allocator {
bool freeHostResource(Pod* pod, consumed_resource_t* consumed, Builder* builder){
	if(host==NULL) {
		std::cerr << " Free Host Resource received NULL host\n";
		exit(2);
	}

	Host* host=pod->getHost();

	//Update the host resources
	host->removePod(pod);
	host->removeAllocaredResource();

	//Need to find the group that has this host and update their resource
	std::vector<Host*> groups = builder->getClusterHosts();
	for(size_t i=0; i<groups.size(); i++) {
		if(builder->findHostInGroup(groups[i]->getId(),host->getId())) {
			groups[i]->removePod(pod);
			break;
		}
	}

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

bool freeHostResource(Task* task, consumed_resource_t* consumed, Builder* builder){
	if(host==NULL) {
		std::cerr << " Free Host Resource received NULL host\n";
		exit(2);
	}

	Pod** pods = task->getPods();
	for(size_t i=0; i< task->getPodsSize(); i++) {
		Host* host=pod->getHost();

		//Update the host resources
		host->removePod(pod);
		host->removeAllocaredResource();

		//Need to find the group that has this host and update their resource
		std::vector<Host*> groups = builder->getClusterHosts();
		for(size_t i=0; i<groups.size(); i++) {
			if(builder->findHostInGroup(groups[i]->getId(),host->getId())) {
				groups[i]->removePod(pod);
				break;
			}
		}

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
	}
	return true;
}
}
#endif
