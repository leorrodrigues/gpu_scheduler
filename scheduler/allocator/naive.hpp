#ifndef _NAIVE_ALLOCATION_
#define _NAIVE_ALLOCATION_

#include "free.hpp"
#include "utils.hpp"

namespace Allocator {

bool naive(Builder* builder,  Task* task, consumed_resource_t* consumed, int current_time){
	spdlog::debug("Naive - Init [{},{}]",current_time, task->getDeadline());
	unsigned int* result = NULL;
	unsigned int resultSize = 0;

	spdlog::debug("Naive - run rank [ ]");
	builder->runRank( builder->getHosts(), current_time, task->getDeadline() );
	spdlog::debug("Naive - run rank [x]");

	spdlog::debug("Naive - get rank result[ ]");
	result = builder->getRankResult(resultSize);
	spdlog::debug("Naive - get rank result[x]");

	//With the hosts ranking made, we iterate through the pods in the specific task
	Host* host=NULL;

	spdlog::debug("Naive - get Task Pods[ ]");
	Pod** pods = task->getPods();
	spdlog::debug("Naive - get Task Pods[x]");

	unsigned int pods_size = task->getPodsSize();
	int interval_low = 0, interval_high = 0;
	size_t pod_index = 0;
	bool pod_allocated;

	size_t i, host_index;
	spdlog::debug("Naive - Iterating through the time {} until {}", current_time, task->getDeadline());
	for(interval_low = current_time; interval_low < task->getDeadline(); interval_low++) {
		interval_high = interval_low + task->getDuration();
		for(pod_index=0; pod_index < pods_size; pod_index++) {
			spdlog::debug("Naive - Will Iterate through pods {}",pod_index);
			pod_allocated = false;
			host=NULL;
			for(host_index = 0; host_index < resultSize; host_index++ ) {
				spdlog::debug("Naive - Will Iterate through hosts {}",host_index);
				spdlog::debug("Naive - get host");
				host=builder->getHost(result[host_index]);
				spdlog::debug("Naive - get host[x]");

				spdlog::debug("Naive - check if pod fit in host");
				if(!checkFit(host,pods[pod_index], interval_low, interval_high))
					continue;
				spdlog::debug("Naive - check if pod fit in host[x]");
				spdlog::debug("Naive - The select host is: {}", host->getId());

				std::map<std::string,std::vector<float> > p_r = pods[pod_index]->getResources();
				host->addPod(interval_low, interval_high, p_r);

				pods[pod_index]->setHost(host);
				pod_allocated=true;
				break;
			}
			//need to desalocate all the allocated pods.
			if(!pod_allocated) {
				for(i = 0; i < pod_index; i++) {
					freeHostResource(pods[i], builder, interval_low, interval_high);
				}
				break;
			}
		}
		if(pod_allocated) break;
	}
	free(result);
	result = NULL;
	if(!pod_allocated)
		return false;
	task->setAllocatedTime(interval_low);
	task->addDelay(interval_low - current_time);
	return true;
}

}
#endif
