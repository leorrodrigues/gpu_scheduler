#ifndef _FIRST_FIT_NOT_INCLUDED_
#define _FIRST_FIT_NOT_INCLUDED_

#include <queue>

#include "../free.hpp"
#include "../utils.hpp"

namespace Allocator {

bool firstFit (Builder* builder,  Task* task,consumed_resource_t* consumed, int current_time){
	std::vector<Host*> hosts = builder->getHosts();

	Pod** pods = task->getPods();
	unsigned int pods_size = task->getPodsSize();
	bool pod_allocated;

	int interval_low = 0, interval_high = 0;
	bool fit = false;
	size_t pod_index = 0;

	for(interval_low = current_time; interval_low < task->getDeadline(); interval_low++) {
		interval_high = interval_low + task->getDuration();
		for(pod_index = 0; pod_index < pods_size; pod_index++) {
			pod_allocated = false;
			for(size_t i=0; i<hosts.size(); i++ ) {
				fit = false;

				if(checkFit(hosts[i], pods[pod_index], interval_low, interval_low + task->getDuration()))
					fit = true;

				if(!fit) continue;

				std::map<std::string,std::vector<float> > p_r = pods[pod_index]->getResources();
				hosts[i]->addPod(interval_low, interval_high, p_r);

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
				for(size_t i=0; i< pod_index; i++) {
					freeHostResource(pods[i],consumed,builder, interval_low, interval_high);
				}
				break;
			}
		}
		if(pod_allocated) break;
	}
	if(!pod_allocated)
		return false;
	task->addDelay(interval_low - current_time);
	return true;
}

}
#endif
