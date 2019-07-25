#ifndef _MULTICRITERIA_CLUSTERIZED_ALLOCATION_
#define _MULTICRITERIA_CLUSTERIZED_ALLOCATION_

#include "free.hpp"
#include "utils.hpp"

namespace Allocator {

// Run the group one time and all the others executions are only with the MULTICRITERIA
bool multicriteria_clusterized(Builder* builder,  Task* task, consumed_resource_t* consumed){

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
	Pod** pods = task->getPods();
	unsigned int pods_size = task->getPodsSize();
	bool pod_allocated;

	//Iterate through the groups and explode each of them
	size_t i=0, j=0, group_index=0;

	std::vector<Host*> groups = builder->getClusterHosts();

	for(size_t pod_index=0; pod_index < pods_size; pod_index++) {

		for( i=0; i<resultSize; i++) {
			pod_allocated = false;

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
			if(!checkFit(groups[group_index], pods[pod_index])) continue;

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
				if(!checkFit(host,pods[pod_index])) continue;

				std::map<std::string,std::vector<float> > p_r = pods[pod_index]->getResources();
				host->addPod(p_r);

				if(!host->getActive()) {
					host->setActive(true);
					consumed->active_servers++;
				}

				addToConsumed(consumed,pods[pod_index]);

				// Update the allocated tasks map
				pods[pod_index]->setHost(host);

				pod_allocated=true;
				break;
			}
			free(ranked_hosts);
			ranked_hosts = NULL;

			if(pod_allocated)
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

	// If didn't has one group and host to support the request, return a false signal
	return true;
}

}
#endif
