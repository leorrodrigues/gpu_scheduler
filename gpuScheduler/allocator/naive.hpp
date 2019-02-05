#ifndef _NAIVE_ALLOCATION_
#define _NAIVE_ALLOCATION_

#include <iostream>
#include <string>
#include <map>

#include "utils.hpp"

namespace Allocator {

bool naive(Builder* builder,  Pod* pod, std::map<unsigned int, unsigned int> &allocated_task,consumed_resource_t* consumed){

	unsigned int* result = NULL;
	unsigned int resultSize = 0;

	builder->runMulticriteria( builder->getHosts() );

	result = builder->getMulticriteriaResult(resultSize);

	Host* host=NULL;

	int i=0;
	printf("Start check hosts\n");
	for( i=0; i<resultSize; i++ ) {
		printf("\t%d of %d\n",i, resultSize);
		host=builder->getHost(result[i]);

		printf("Host get\n");
		int fit=checkFit(host,pod);

		printf("Check Fit made\n");
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

		allocated_task[pod->getId()]= host->getId();

		free(result);
		result=NULL;

		printf("Found\n");
		return true;
	}
	free(result);
	result=NULL;

	printf("returning null\n");
	return false;
}

}
#endif
