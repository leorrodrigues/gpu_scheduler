#ifndef _FIRST_FIT_NOT_INCLUDED_
#define _FIRST_FIT_NOT_INCLUDED_

#include <queue>

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

			if(!checkFit(hosts[i],pods[pod_index])) continue;

			std::map<std::string,std::tuple<float,float,bool> > p_r = pods[pod_index]->getResources();
			hosts[i]->addPod(p_r);

			if(!hosts[i]->getActive()) {
				hosts[i]->setActive(true);
				consumed->active_servers++;
			}

			addToConsumed(consumed,pods[pod_index]);

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
