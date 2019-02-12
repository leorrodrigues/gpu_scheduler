#ifndef _FIRST_FIT_NOT_INCLUDED_
#define _FIRST_FIT_NOT_INCLUDED_

#include <iostream>
#include <queue>
#include <string>
#include <map>

#include "../utils.hpp"

namespace Allocator {

bool firstFit (Builder* builder,  Task* task, std::map<unsigned int, unsigned int> &allocated_task,consumed_resource_t* consumed){
	std::vector<Host*> hosts = builder->getHosts();
	int i=0;

	for( i=0; i<hosts.size(); i++ ) {
		int fit=checkFit(hosts[i],pod);
		if(fit==0) {
			continue;
		}

		pod->setFit(fit);
		hosts[i]->addPod(pod);

		if(hosts[i]->getActive()==false) {
			hosts[i]->setActive(true);
			consumed->active_servers++;
		}

		// The pod was allocated, so the consumed variable has to be updated
		if(fit==7) { // allocate MAX VCPU AND RAM
			consumed->ram  += pod->getRamMax();
			consumed->vcpu += pod->getVcpuMax();
		}else if(fit==8) { // ALLOCATE MAX VCPU AND RAM MIN
			consumed->ram  += pod->getRamMin();
			consumed->vcpu += pod->getVcpuMax();
		}else if(fit==10) { // ALLOCATE VCPU MIN AND RAM MAX
			consumed->ram  += pod->getRamMax();
			consumed->vcpu += pod->getVcpuMin();
		}else if(fit==11) { // ALLOCATE VCPU AND RAM MIN
			consumed->ram  += pod->getRamMin();
			consumed->vcpu += pod->getVcpuMin();
		}


		hosts[i]->addAllocatedResources();

		allocated_task[pod->getId()]= hosts[i]->getId();

		return true;
	}
	return false;
}

};
#endif
