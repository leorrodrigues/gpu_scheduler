#ifndef _NAIVE_ALLOCATION_
#define _NAIVE_ALLOCATION_

#include <iostream>
#include <string>
#include <map>

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

	for(size_t pod_index; pod_index < pods_size; pod_index++) {
		pod_success = false;
		host=NULL;
		for(size_t host_index=0; host_index<resultSize; host_index++ ) {
			host=builder->getHost(result[host_index]);

			int fit=checkFit(host,pods[pod_index]);

			if(fit==0) {
				// If can't ignore the rest of the loop
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


			pod_sucess=true;
		}

		if(pod_sucess==false) {
			//need to desalocate all the allocated pods.
			for(size_t i=0; i< pod_index; i++)
				freeHostResource(pods[i]->getHost(),consumed,builder);

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
