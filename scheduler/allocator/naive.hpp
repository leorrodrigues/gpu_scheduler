#ifndef _NAIVE_ALLOCATION_
#define _NAIVE_ALLOCATION_

#include "free.hpp"
#include "utils.hpp"

namespace Allocator {

bool naive(Builder* builder,  Task* task, consumed_resource_t* consumed){
	// printf("Try to allocate TASK %d\n",task->getId());
	unsigned int* result = NULL;
	unsigned int resultSize = 0;

	// printf("Running multicriteria\n");
	builder->runMulticriteria( builder->getHosts() );

	// printf("Get Multicriteria Result\n");
	result = builder->getMulticriteriaResult(resultSize);

	//With the hosts ranking made, we iterate through the pods in the specific task
	Host* host=NULL;
	// printf("Get Task Pods\n");
	Pod** pods = task->getPods();
	unsigned int pods_size = task->getPodsSize();
	bool pod_allocated;

	// printf("Will Iterate through pods\n");
	for(size_t pod_index=0; pod_index < pods_size; pod_index++) {
		pod_allocated = false;
		host=NULL;
		for(size_t host_index=0; host_index<resultSize; host_index++ ) {
			// printf("-----------------------------------------\n");
			host=builder->getHost(result[host_index]);

			if(!checkFit(host,pods[pod_index])) continue;

			std::map<std::string,std::tuple<float,float,bool> > p_r = pods[pod_index]->getResources();
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
