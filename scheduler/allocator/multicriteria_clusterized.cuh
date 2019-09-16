#ifndef _MULTICRITERIA_CLUSTERIZED_ALLOCATION_
#define _MULTICRITERIA_CLUSTERIZED_ALLOCATION_

#include "free.hpp"
#include "utils.hpp"

namespace Allocator {

// Run the group one time and all the others executions are only with the MULTICRITERIA
bool multicriteria_clusterized(Builder* builder,  Task* task, consumed_resource_t* consumed, int current_time){
	spdlog::debug("Multicriteria Clusterized - Init [{},{}]",current_time, task->getDeadline());
	// Create the result variables
	unsigned int* result = NULL;
	unsigned int resultSize = 0;

	// Create the clustered variables
	unsigned int* ranked_hosts = NULL;
	unsigned int ranked_hosts_size = 0;

	// If the number of groups are bigger than 1, the multicriteria method has run to select the most suitable group
	spdlog::debug("Multicriteria Clusterized - run multicriteria[ ]");
	builder->runRankClustered( builder->getClusterHosts(), current_time, task->getDeadline());
	spdlog::debug("Multicriteria Clusterized - run multicriteria[x]");

	// Get the results
	spdlog::debug("Multicriteria Clusterized - get multicriteria result[ ]");
	result = builder->getRankClusteredResult(resultSize);
	spdlog::debug("Multicriteria Clusterized - get multicriteria result[x]");

	// After the MULTICRITERIA_CLUSTERIZED groups are made and the Multicriteria method selected the most suitable group in the MULTICRITERIA_CLUSTERIZED, the selected group is opened ah their hosts selected by the multicriteria method to select the host for the request.
	// Create the empty host
	Host* host=NULL;
	Pod** pods = task->getPods();
	unsigned int pods_size = task->getPodsSize();
	bool pod_allocated;

	//Iterate through the groups and explode each of them
	size_t i = 0, j = 0, group_index = 0, pod_index = 0;

	spdlog::debug("Multicriteria Clusterized - getClusterHosts[ ]");
	std::vector<Host*> groups = builder->getClusterHosts();
	spdlog::debug("Multicriteria Clusterized - getClusterHosts[x]");

	int interval_low = 0, interval_high = 0;
	bool fit = false;

	spdlog::debug("Multicriteria Clusterized - time loop[ ]");
	for(interval_low = current_time; interval_low < task->getDeadline(); interval_low++) {
		interval_high = interval_low + task->getDuration();

		spdlog::debug("Multicriteria Clusterized - pod loop[ ]");
		for(pod_index=0; pod_index < pods_size; pod_index++) {

			spdlog::debug("Multicriteria Clusterized - group loop[ ]");
			for( i=0; i<resultSize; i++) {
				pod_allocated = false;
				fit = false;

				host=NULL;
				//Get the group Index
				spdlog::debug("Multicriteria Clusterized - group loop - get index[ ]");
				for(group_index=0; group_index < groups.size(); group_index++) {
					if(groups[group_index]->getId()==result[i]) {
						break;
					}
				}
				spdlog::debug("Multicriteria Clusterized - group loop - get index[x]");

				spdlog::debug("Multicriteria Clusterized - group loop - check fit[ ]");
				if(checkFit(groups[group_index],pods[pod_index], interval_low, interval_high))
					fit = true;
				spdlog::debug("Multicriteria Clusterized - group loop - check fit[x]");

				if(!fit) continue;

				std::vector<Host*> hostsInGroup = builder->getHostsInGroup(result[i]);

				// Run the Multicriteria in the hosts
				spdlog::debug("Multicriteria Clusterized - group loop - run mcdm[ ]");
				builder->runRank(hostsInGroup, interval_low, interval_high);
				spdlog::debug("Multicriteria Clusterized - group loop - run mcdm[ ]");

				// Get the result
				ranked_hosts_size = 0;
				spdlog::debug("Multicriteria Clusterized - group loop - get mcdm[ ]");
				ranked_hosts = builder->getRankResult(ranked_hosts_size);
				spdlog::debug("Multicriteria Clusterized - group loop - get mcdm[ ]");

				// Iterate through all the hosts in the selected group
				spdlog::debug("Multicriteria Clusterized - hosts loop[ ]");
				for( j=0; j<ranked_hosts_size; j++) {
					fit = false;
					// Get the host pointer
					host=builder->getHost(ranked_hosts[j]);
					// Check if the host can support the resource

					if(checkFit(host,pods[pod_index], interval_low, interval_high))
						fit = true;

					if(!fit) continue;

					std::map<std::string,std::vector<float> > p_r = pods[pod_index]->getResources();
					host->addPod(interval_low, interval_high, p_r);

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
				spdlog::debug("Multicriteria Clusterized - hosts loop[x]");

				free(ranked_hosts);
				ranked_hosts = NULL;

				if(pod_allocated)
					break;
			}
			spdlog::debug("Multicriteria Clusterized - groups loop[x]");

			if(!pod_allocated) {
				//need to desalocate all the allocated pods.
				for(size_t i=0; i< pod_index; i++) {
					freeHostResource(pods[i],consumed,builder, interval_low, interval_high);
				}
				break;
			}
		}
		spdlog::debug("Multicriteria Clusterized - pod loop[x]");
		if(pod_allocated) break;
	}
	spdlog::debug("Multicriteria Clusterized - time loop[x]");
	free(result);
	result=NULL;
	if(!pod_allocated)
		return false;
	//set the delay applied to the request
	task->addDelay(interval_low - current_time);
	return true;
}

}
#endif
