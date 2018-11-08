#ifndef _DC_ALLOCATION_
#define _DC_ALLOCATION_

#include <iostream>

#include "utils.hpp"

namespace Allocator {

bool dc(Builder* builder,  Container* container, std::map<int,char*> &allocated_task){
	// std::cout << "Running Clustering\n";
	// Now the MCL is used to cluster the DC
	builder->runClustering(builder->getHosts());

	// std::cout << "Cluster ok\n";
	// std::cout << "Getting Cluster Result\n";
	// std::cout << "Groups Made "<<builder->getClusteringResultSize()<<"\n";

	// Get the cluster results and update the builder
	// std::cout << "Cluster Results ok\n";
	builder->getClusteringResult();

	// // Create the result map
	// std::map<int,std::string> result;
	//
	// // If the number of groups are made, the multicriteria method has run to select the most suitable group
	// if(builder->getClusteringResultSize()>1) {
	//      // std::cout << "Running Multicriteria\n";
	//      // Run the multicriteria with the cluster
	//      builder->runMulticriteria( builder->getClusterHosts() );
	//
	//      // Get the results
	//      result = builder->getMulticriteriaResult();
	//      // std::cout << "Multicriteria OK\n";
	// }else{
	//      // Create the first entry in the result map
	//      result[0]="0";
	// }
	//
	// // After the DC groups are made and the Multicriteria method selected the most suitable group in the DC, the selected group is opened ah their hosts selected by the multicriteria method to select the host for the request.
	// // Create the empty host
	// Host* host=NULL;
	//
	// //Iterate through the groups and explode each of them
	// // std::cout << "Start the iteration\n";
	// for( std::map<int,std::string>::iterator it = result.begin(); it!=result.end(); it++) {
	//      host=NULL;
	//      // std::cout<<it->first<<" AND "<<it->second<<"\n";
	//      std::vector<Host*> hostsInGroup = builder->getHostsInGroup(std::stoi(it->second));
	//      // std::cout<<"Running host multicriteria\n";
	//      // Run the Multicriteria in the hosts
	//      builder->runMulticriteria(hostsInGroup);
	//      // std::cout<<"Get the multicriteria result\n";
	//      // Get the result
	//      std::map<int,std::string> ranked_hosts = builder->getMulticriteriaResult();
	//      // std::cout<<"start iteration\n";
	//      // Iterate through all the hosts in the selected group
	//      for(std::map<int,std::string>::iterator h_it = ranked_hosts.begin(); h_it!= ranked_hosts.end(); h_it++) {
	//              // Get the host pointer
	//              // std::cout<<"iterating host\n";
	//
	//              host=builder->getHost(h_it->second);
	//
	//              // Check if the host can support the resource
	//              if(!checkFit(host,container)) {
	//                      // If can't ignore the rest of the loop
	//                      continue;
	//              }
	//              // If can, allocate
	//              (*host)-=(*container);
	//              // Update the allocated tasks map
	//              allocated_task[container->getId()]=host->getName();
	//              // std::cout<<"Allocated!\n";
	//              // End the function with true signal
	//              return true;
	//      }
	// }
	// If didn't has one group and host to support the request, return a false signal
	// return false;
	return true;
}

}
#endif
