#ifndef _MULTICRITERIA_CLUSTERIZED_ALLOCATION_
#define _MULTICRITERIA_CLUSTERIZED_ALLOCATION_

#include <iostream>

#include "utils.hpp"

namespace Allocator {

// Run the group one time and all the others executions are only with the MULTICRITERIA
bool multicriteria_clusterized(Builder* builder,  Task* task, std::map<unsigned int, unsigned int> &allocated_task, consumed_resource_t* consumed){

	// Create the result variables
	unsigned int* result = NULL;
	unsigned int resultSize = 0;

	// Create the clustered variables
	unsigned int* ranked_hosts = NULL;
	unsigned int ranked_hosts_size = 0;

	// If the number of groups are bigger than 1, the multicriteria method has run to select the most suitable group
	builder->runMulticriteriaClustered( builder->getClusterHosts() );

	// Get the results
	result = builder->getMulticriteriaClusteredResult(resultSize);

	// After the MULTICRITERIA_CLUSTERIZED groups are made and the Multicriteria method selected the most suitable group in the MULTICRITERIA_CLUSTERIZED, the selected group is opened ah their hosts selected by the multicriteria method to select the host for the request.
	// Create the empty host
	Host* host=NULL;

	//Iterate through the groups and explode each of them
	int i=0, j=0, group_index=0;

	std::vector<Host*> groups = builder->getClusterHosts();

	for( i=0; i<resultSize; i++) {
		host=NULL;
		// for(group_index=0; group_index < groups.size(); group_index++) {
		// printf("GROUP %d VCPU %f RAM %f\n",groups[group_index]->getId(), groups[group_index]->getResource()["vcpu"], groups[group_index]->getResource()["memory"]);
		// }
		// printf("GROUP %d POSITION %d\n",result[i],i);
		// getchar();

		//Get the group Index
		for(group_index=0; group_index < groups.size(); group_index++) {
			if(groups[group_index]->getId()==result[i]) {
				break;
			}
		}

		//If the group can't contain the pod, search the next group
		if(checkFit(groups[group_index], pod)==0) {
			continue;
		}
		std::vector<Host*> hostsInGroup = builder->getHostsInGroup(result[i]);

		// Run the Multicriteria in the hosts
		builder->runMulticriteria(hostsInGroup);

		// Get the result
		ranked_hosts_size = 0;
		ranked_hosts = builder->getMulticriteriaResult(ranked_hosts_size);

		// Iterate through all the hosts in the selected group
		for( j=0; j<ranked_hosts_size; j++) {
			// Get the host pointer
			host=builder->getHost(ranked_hosts[j]);
			// Check if the host can support the resource
			int fit=checkFit(host,pod);

			if(fit==0) {
				// If can't ignore the rest of the loop
				continue;
			}

			pod->setFit(fit);
			//update the host resources
			host->addPod(pod);
			//update the group resources
			groups[group_index]->addPod(pod);

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
			// Update the allocated tasks map
			allocated_task[pod->getId()]= host->getId();

			free(ranked_hosts);
			ranked_hosts = NULL;

			free(result);
			result=NULL;

			// End the function with true signal
			return true;
		}
		free(ranked_hosts);
		ranked_hosts = NULL;
	}
	free(result);
	result=NULL;
	// If didn't has one group and host to support the request, return a false signal
	return false;
}

}
#endif
