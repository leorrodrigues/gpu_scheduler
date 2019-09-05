#ifndef _FIRST_FIT_NOT_INCLUDED_
#define _FIRST_FIT_NOT_INCLUDED_

#include <queue>

#include "../free.hpp"
#include "../utils.hpp"

namespace Allocator {

bool firstFit (Builder* builder,  Task* task,consumed_resource_t* consumed, int interval_low){
	std::vector<Host*> hosts = builder->getHosts();

	Pod** pods = task->getPods();
	unsigned int pods_size = task->getPodsSize();
	bool pod_allocated;

	int interval_high = 0;
	bool fit = false;
	for(size_t pod_index=0; pod_index < pods_size; pod_index++) {
		for(size_t i=0; i<hosts.size(); i++ ) {
			fit = false;

			for(interval_high = interval_low+1; interval_high < task->getDeadLine(); interval_high++) {
				if(checkFit(hosts[i],pods[pod_index], interval_low, interval_high)) {
					fit = true;
					break;
				}

			}
			if(!fit) continue;

			std::map<std::string,std::vector<float> > p_r = pods[pod_index]->getResources();
			hosts[i]->addPod(interval_low, interval_high, p_r);

			if(!hosts[i]->getActive()) {
				hosts[i]->setActive(true);
				consumed->active_servers++;
			}

			addToConsumed(consumed,pods[pod_index], interval_low, interval_high);

			pods[pod_index]->setHost(hosts[i]);

			pod_allocated=true;
			break;
		}
		if(!pod_allocated) {
			//need to desalocate all the allocated pods.
			for(size_t i=0; i< pod_index; i++)
				freeHostResource(pods[i],consumed,builder, interval_low, interval_high);
			return false;
		}
	}
	return true;
}

};
#endif
