#ifndef _DC_ALLOCATION_
#define _DC_ALLOCATION_

#include <iostream>

#include "utils.hpp"

namespace Allocator {

bool dc(Builder* builder,  Container* container, std::map<unsigned int, unsigned int> &allocated_task,consumed_resource_t* consumed){

	// Create the result variables
	unsigned int resultSize = 0;
	unsigned int* result = NULL;

	// Create the result clustered variables
	unsigned int ranked_hosts_size = 0;
	unsigned int* ranked_hosts = NULL;

	// Now the MCL is used to cluster the DC
	builder->runClustering(builder->getHosts());

	// Get the cluster results and update the builder
	// std::cout << "Cluster Results ok\n";
	builder->getClusteringResult();

	// Run the multicriteria with the cluster
	builder->runMulticriteria( builder->getClusterHosts() );

	// Get the results
	result = builder->getMulticriteriaResult(resultSize);

	// After the DC groups are made and the Multicriteria method selected the most suitable group in the DC, the selected group is opened ah their hosts selected by the multicriteria method to select the host for the request.
	// Create the empty host
	Host* host=NULL;

	//Iterate through the groups and explode each of them
	// std::cout << "Start the iteration\n";
	int i=0, j=0;
	for( i=0; i<resultSize; i++) {
		host=NULL;

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

			// Update the allocated tasks map
			allocated_task[container->getId()]= host->getId();

			free(ranked_hosts);
			ranked_hosts = NULL;
			free(result);
			result = NULL;

			// End the function with true signal
			return true;
		}

		free(ranked_hosts);
		ranked_hosts = NULL;
	}

	free(result);
	result = NULL;

	// If didn't has one group and host to support the request, return a false signal
	return false;
}

}
#endif
