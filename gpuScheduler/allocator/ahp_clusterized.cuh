#ifndef _AHP_CLUSTERIZED_ALLOCATION_
#define _AHP_CLUSTERIZED_ALLOCATION_

#include <iostream>

#include "utils.hpp"

namespace Allocator {

// Run the group one time and all the others executions are only with the AHP
bool ahp_clusterized(Builder* builder,  Container* container, std::map<int,const char*> &allocated_task,consumed_resource_t* consumed){
	// printf("##################################################\n");
	// std::cout << "\t\tRunning Clustering\n";
	// If the AHP_CLUSTERIZED is not clusterized
	if(builder->getClusterHosts().size()==0) {
		// Now the MCL is used to cluster the AHP_CLUSTERIZED
		builder->runClustering(builder->getHosts());
		builder->getClusteringResult();
	}
	// std::cout << "\t\tCluster ok\n";
	// std::cout << "\t\tGroups Made "<<builder->getClusteringResultSize()<<"\n";

	// Get the cluster results and update the builder
	// std::cout << "\t\tCluster Results ok\n";
	// printf("\t\tDone\n");
	// Create the result map
	std::map<int,const char*> result;

	// If the number of groups are made, the multicriteria method has run to select the most suitable group
	if(builder->getClusteringResultSize()>1) {
		// std::cout << "Running Multicriteria\n";
		// Run the multicriteria with the cluster
		builder->runMulticriteriaClustered( builder->getClusterHosts() );

		// Get the results
		result = builder->getMulticriteriaClusteredResult();
		// std::cout << "Multicriteria OK\n";
	}else{
		// Create the first entry in the result map
		result[0]="0";
	}

	// After the AHP_CLUSTERIZED groups are made and the Multicriteria method selected the most suitable group in the AHP_CLUSTERIZED, the selected group is opened ah their hosts selected by the multicriteria method to select the host for the request.
	// Create the empty host
	Host* host=NULL;

	//Iterate through the groups and explode each of them
	// std::cout << "Start the iteration\n";
	for( auto const& it: result) {
		host=NULL;
		// std::cout<<it->first<<" AND "<<it->second<<"\n";
		std::vector<Host*> hostsInGroup = builder->getHostsInGroup(std::stoi(it.second));
		// std::cout<<"Running host multicriteria\n";
		// Run the Multicriteria in the hosts
		builder->runMulticriteria(hostsInGroup);
		// std::cout<<"Get the multicriteria result\n";
		// Get the result

		std::map<int,const char*> ranked_hosts = builder->getMulticriteriaResult();
		// Iterate through all the hosts in the selected group
		for(std::map<int,const char*>::iterator h_it = ranked_hosts.begin(); h_it!= ranked_hosts.end(); h_it++) {
			// Get the host pointer

			host=builder->getHost(std::string(h_it->second));
			// Check if the host can support the resource
			int fit=checkFit(host,container);
			if(fit==0) {
				// If can't ignore the rest of the loop
				continue;
			}else{
				container->setFit(fit);
				host->addContainer(container);
			}

			if(host->getActive()==false) {
				host->setActive(true);
				consumed->active_servers++;
			}

			host->addAllocatedResources();
			// Update the allocated tasks map
			char* host_name = (char*) malloc (strlen(host->getName().c_str())+1);
			strcpy(host_name,host->getName().c_str());

			allocated_task[container->getId()]= &host_name[0];

			// std::cout<<"\t\tAllocated!\n";
			// End the function with true signal
			return true;
		}
	}
	// printf("\t\tError in Allocated!\n");
	// If didn't has one group and host to support the request, return a false signal
	return false;
}

}
#endif
