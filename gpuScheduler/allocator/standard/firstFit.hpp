#ifndef _FIRST_FIT_NOT_INCLUDED_
#define _FIRST_FIT_NOT_INCLUDED_

#include <iostream>
#include <string>
#include <queue>
#include <map>

#include "../free.hpp"
#include "../utils.hpp"

namespace Allocator {

bool firstFit (Builder* builder,  Task* task,consumed_resource_t* consumed){
	std::vector<Host*> hosts = builder->getHosts();

	Pod** pods = task->getPods();
	unsigned int pods_size = task->getPodsSize();
	bool pod_allocated;

	for(size_t pod_index=0; pod_index < pods_size; pod_index++) {
		for(size_t i=0; i<hosts.size(); i++ ) {

			int fit=checkFit(hosts[i],pods[pod_index]);
			if(fit==0) {
				continue;
			}

			pods[pod_index]->setFit(fit);
			std::map<std::string,float> p_r = pods[pod_index]->getResources();
			hosts[i]->addPod(p_r, fit);

			if(!hosts[i]->getActive()) {
				hosts[i]->setActive(true);
				consumed->active_servers++;
			}

			addToConsumed(consumed,p_r,fit);

			hosts[i]->addAllocatedResources();

			pods[pod_index]->setHost(hosts[i]);

			pod_allocated=true;
			break;
		}
		if(!pod_allocated) {
			//need to desalocate all the allocated pods.
			for(size_t i=0; i< pod_index; i++)
				freeHostResource(pods[i],consumed,builder);
			return false;
		}
	}
	return true;
}

};
#endif
