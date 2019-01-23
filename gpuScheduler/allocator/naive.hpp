#ifndef _NAIVE_ALLOCATION_
#define _NAIVE_ALLOCATION_

#include <iostream>
#include <string>
#include <map>

#include "../datacenter/tasks/container.hpp"
#include "../builder.cuh"
#include "utils.hpp"

namespace Allocator {

bool naive(Builder* builder,  Container* container, std::map<unsigned int, unsigned int> &allocated_task,consumed_resource_t* consumed){
	unsigned int* result = NULL;
	unsigned int resultSize = 0;

	builder->runMulticriteria( builder->getHosts() );

	result = builder->getMulticriteriaResult(resultSize);

	Host* host=NULL;

	int i=0;
	for( i=0; i<resultSize; i++ ) {
		host=builder->getHost(result[i]);

		int fit=checkFit(host,container);
		if(fit==0) {
			// If can't ignore the rest of the loop
			continue;
		}

		container->setFit(fit);
		host->addContainer(container);

		if(host->getActive()==false) {
			host->setActive(true);
			consumed->active_servers++;
		}

		// The container was allocated, so the consumed variable has to be updated
		if(fit==7) { // allocate MAX VCPU AND RAM
			consumed->ram += container->containerResources->ram_max;
			consumed->vcpu +=container->containerResources->vcpu_max;
		}else if(fit==8) { // ALLOCATE MAX VCPU AND RAM MIN
			consumed->ram += container->containerResources->ram_min;
			consumed->vcpu += container->containerResources->vcpu_max;
		}else if(fit==10) { // ALLOCATE VCPU MIN AND RAM MAX
			consumed->ram += container->containerResources->ram_max;
			consumed->vcpu +=container->containerResources->vcpu_min;
		}else if(fit==11) { // ALLOCATE VCPU AND RAM MIN
			consumed->ram += container->containerResources->ram_min;
			consumed->vcpu += container->containerResources->vcpu_min;
		}

		host->addAllocatedResources();

		allocated_task[container->getId()]= host->getId();

		free(result);
		result=NULL;

		return true;
	}
	free(result);
	result=NULL;

	return false;
}

}
#endif
