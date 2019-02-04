#ifndef _WORST_FIT_NOT_INCLUDED_
#define _WORST_FIT_NOT_INCLUDED_

#include <iostream>
#include <queue>
#include <string>
#include <map>

#include "../utils.hpp"

struct CompareWF {
	bool operator()(Host* lhs, Host* rhs) const {
		float r1=0, r2=0;
		for(std::map<std::string,float>::iterator it_1 = lhs->getResource().begin(), it_2 = rhs->getResource().begin(); it_1!=lhs->getResource().end(); it_1++, it_2++) {
			r1+=it_1->second;
			r2+=it_2->second;
		}
		if(r1>r2) return true;
		else return false;
	}
};

namespace Allocator {

bool worstFit(Builder* builder,  Container* container, std::map<unsigned int, unsigned int> &allocated_task,consumed_resource_t* consumed){
	std::vector<Host*> aux = builder->getHosts();
	std::priority_queue<Host*, std::vector<Host*>, CompareWF> hosts (aux.begin(), aux.end());
	Host* host = NULL;

	while(!hosts.empty()) {
		host =  hosts.top();
		hosts.pop();

		int fit=checkFit(host,container);
		if(fit==0) {
			continue;
		}

		container->setFit(fit);
		host->addContainer(container);

		if(host->getActive()==false) {
			host->setActive(true);
			consumed->active_servers++;
		}

		// The container was allocated, so the consumed variable has to be updated
		if(fit==7) { // allocate MAX VCPU AND RAM
			consumed->ram += container->containerResources->ram_max;
			consumed->vcpu +=container->containerResources->vcpu_max;
		}else if(fit==8) { // ALLOCATE MAX VCPU AND RAM MIN
			consumed->ram += container->containerResources->ram_min;
			consumed->vcpu += container->containerResources->vcpu_max;
		}else if(fit==10) { // ALLOCATE VCPU MIN AND RAM MAX
			consumed->ram += container->containerResources->ram_max;
			consumed->vcpu +=container->containerResources->vcpu_min;
		}else if(fit==11) { // ALLOCATE VCPU AND RAM MIN
			consumed->ram += container->containerResources->ram_min;
			consumed->vcpu += container->containerResources->vcpu_min;
		}

		host->addAllocatedResources();

		allocated_task[container->getId()]= host->getId();

		return true;
	}
	return false;
}

}
#endif
