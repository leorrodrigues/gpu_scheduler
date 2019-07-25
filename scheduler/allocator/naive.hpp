#ifndef _NAIVE_ALLOCATION_
#define _NAIVE_ALLOCATION_

#include "free.hpp"
#include "utils.hpp"

namespace Allocator {

bool naive(Builder* builder,  Task* task, consumed_resource_t* consumed){
	spdlog::debug("Naive - Init");
	// printf("Try to allocate TASK %d\n",task->getId());
	unsigned int* result = NULL;
	unsigned int resultSize = 0;

	// printf("Running multicriteria\n");
	spdlog::debug("Naive - run multicriteria");
	builder->runMulticriteria( builder->getHosts() );
	spdlog::debug("Naive - run multicriteria[x]");

	// printf("Get Multicriteria Result\n");
	spdlog::debug("Naive - get multicriteria result");
	result = builder->getMulticriteriaResult(resultSize);
	spdlog::debug("Naive - get multicriteria result[x]");

	//With the hosts ranking made, we iterate through the pods in the specific task
	Host* host=NULL;
	// printf("Get Task Pods\n");

	spdlog::debug("Naive - get pods");
	Pod** pods = task->getPods();
	spdlog::debug("Naive - get pods[x]");

	unsigned int pods_size = task->getPodsSize();
	bool pod_allocated;

	for(size_t pod_index=0; pod_index < pods_size; pod_index++) {
		spdlog::debug("Naive - Will Iterate through pods {}",pod_index);
		pod_allocated = false;
		host=NULL;
		for(size_t host_index=0; host_index<resultSize; host_index++ ) {
			spdlog::debug("Naive - Will Iterate through hosts {}",host_index);
			// printf("-----------------------------------------\n");
			spdlog::debug("Naive - get host");
			host=builder->getHost(result[host_index]);
			spdlog::debug("Naive - get host[x]");

			spdlog::debug("Naive - check if pod fit in host");
			if(!checkFit(host,pods[pod_index])) continue;
			spdlog::debug("Naive - check if pod fit in host[x]");

			std::map<std::string,std::vector<float> > p_r = pods[pod_index]->getResources();
			host->addPod(p_r);

			if(!host->getActive()) {
				host->setActive(true);
				consumed->active_servers++;
			}

			addToConsumed(consumed,pods[pod_index]);

			pods[pod_index]->setHost(host);

			pod_allocated=true;
			break;
		}

		if(!pod_allocated) {
			//need to desalocate all the allocated pods.
			for(size_t i=0; i< pod_index; i++)
				freeHostResource(pods[i],consumed,builder);

			free(result);
			result=NULL;
			return false;
		}

	}

	free(result);
	result=NULL;

	return true;
}

}
#endif
