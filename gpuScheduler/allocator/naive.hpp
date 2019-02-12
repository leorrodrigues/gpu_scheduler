#ifndef _NAIVE_ALLOCATION_
#define _NAIVE_ALLOCATION_

#include <iostream>
#include <string>
#include <map>

#include "free.hpp"
#include "utils.hpp"

namespace Allocator {

bool naive(Builder* builder,  Task* task, consumed_resource_t* consumed){

	unsigned int* result = NULL;
	unsigned int resultSize = 0;

	builder->runMulticriteria( builder->getHosts() );

	result = builder->getMulticriteriaResult(resultSize);

	//With the hosts ranking made, we iterate through the pods in the specific task
	Host* host=NULL;
	Pod** pods = task->getPods();
	unsigned int pods_size = task->getPodsSize();
	bool pod_allocated;

	for(size_t pod_index=0; pod_index < pods_size; pod_index++) {
		pod_allocated = false;
		host=NULL;
		for(size_t host_index=0; host_index<resultSize; host_index++ ) {
			host=builder->getHost(result[host_index]);

			int fit=checkFit(host,pods[pod_index]);

			if(fit==0) {
				// If can't ignore the rest of the loop
				continue;
			}

			pods[pod_index]->setFit(fit);
			std::map<std::string,float> p_r = pods[pod_index]->getResources();
			host->addPod(p_r, fit);

			if(host->getActive()==false) {
				host->setActive(true);
				consumed->active_servers++;
			}

			addToConsumed(consumed,p_r,fit);

			host->addAllocatedResources();

			pod_allocated=true;
		}

		if(pod_allocated==false) {
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
