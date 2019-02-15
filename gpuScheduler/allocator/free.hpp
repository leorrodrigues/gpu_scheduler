#ifndef _FREE_ALLOCATION_
#define _FREE_ALLOCATION_

#include "../datacenter/tasks/pod.hpp"
#include "../datacenter/host.hpp"
#include "utils.hpp"

namespace Allocator {
bool freeHostResource(Pod* pod, consumed_resource_t* consumed, Builder* builder){
	Host* host=pod->getHost();

	//Update the host resources
	host->removePod(pod->getResources());

	//Need to find the group that has this host and update their resource
	std::vector<Host*> groups = builder->getClusterHosts();
	for(size_t i=0; i<groups.size(); i++) {
		if(builder->findHostInGroup(groups[i]->getId(),host->getId())) {
			groups[i]->removePod(pod->getResources());
			break;
		}
	}

	subToConsumed(consumed, pod);

	return true;
}

bool freeHostResource(Task* task, consumed_resource_t* consumed, Builder* builder){
	Pod** pods = task->getPods();
	for(size_t i=0; i< task->getPodsSize(); i++) {
		Host* host=pods[i]->getHost();

		//Update the host resources
		host->removePod(pods[i]->getResources());
		host->removeAllocaredResource();

		//Need to find the group that has this host and update their resource
		std::vector<Host*> groups = builder->getClusterHosts();
		for(size_t j=0; j<groups.size(); j++) {
			if(builder->findHostInGroup(groups[j]->getId(),host->getId())) {
				groups[j]->removePod(pods[i]->getResources());
				break;
			}
		}

		subToConsumed(consumed, pods[i]);
	}
	return true;
}
}
#endif
