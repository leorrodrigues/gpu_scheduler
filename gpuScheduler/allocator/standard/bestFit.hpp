#ifndef _BEST_FIT_NOT_INCLUDED_
#define _BEST_FIT_NOT_INCLUDED_

#include <iostream>
#include <queue>
#include <string>
#include <map>

#include "../utils.hpp"

struct CompareBF {
	bool operator()(Host* lhs, Host* rhs) const {
		float r1=0, r2=0;
		for(std::map<std::string,float>::iterator it_1 = lhs->getResource().begin(), it_2 = rhs->getResource().begin(); it_1!=lhs->getResource().end(); it_1++, it_2++) {
			r1+=it_1->second;
			r2+=it_2->second;
		}
		if(r1<r2) return true;
		else return false;
	}
};

namespace Allocator {
bool bestFit(Builder* builder,  Task* task, std::map<unsigned int, unsigned int> &allocated_task,consumed_resource_t* consumed){
	std::vector<Host*> aux = builder->getHosts();
	std::priority_queue<Host*, std::vector<Host*>, CompareBF> hosts (aux.begin(), aux.end());
	Host* host = NULL;

	while(!hosts.empty()) {
		host =  hosts.top();
		hosts.pop();

		int fit=checkFit(host,pod);
		if(fit==0) {
			continue;
		}

		pod->setFit(fit);
		host->addPod(pod);

		if(host->getActive()==false) {
			host->setActive(true);
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

		host->addAllocatedResources();

		allocated_task[pod->getId()]= host->getId();

		return true;
	}
	return false;
}

}
#endif
