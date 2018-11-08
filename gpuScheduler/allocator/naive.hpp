#ifndef _NAIVE_ALLOCATION_
#define _NAIVE_ALLOCATION_

#include <iostream>
#include <string>
#include <map>

#include "../datacenter/tasks/container.hpp"
#include "../builder.cuh"
#include "utils.hpp"

namespace Allocator {

bool naive(Builder* builder,  Container* container, std::map<int,char*> &allocated_task){
	// std::cout << "##############################\nTry the allocation\n";

	// std::cout << "Running Multicriteria\n";
	// builder->listHosts();

	builder->runMulticriteria( builder->getHosts() );

	// std::cout << "Multicriteria OK\n";
	// std::cout << "Getting Results\n";
	std::map<int,char*> result = builder->getMulticriteriaResult();

	// Para testes
	// std::cout << "Results Found!\nAllocating\n";
	Host* host=NULL;
	for( std::map<int,char*>::iterator it = result.begin(); it!=result.end(); it++) {
		// std::cout<< "Checking the host "<<it->first<<":"<<it->second<<"\n";
		host=builder->getHost(std::string(it->second));
		// std::cout << "HOST RESOURCES\n";
		// std::cout << "VCPU: "<< host->getResource()->mWeight["vcpu"]<<"\n";
		// std::cout << "RAM: "<< host->getResource()->mWeight["memory"] <<"\n";

		if(!checkFit(host,container)) {
			// std::cout<<"This host cant support the container\n";

			continue;
		}

		// std::cout<<"Subtracting the resources\n";
		(*host)-=(*container);
		// std::cout<<"Subtracted\n";
		// std::cout << "HOST RESOURCES 2\n";
		// std::cout << "VCPU: "<< host->getResource()->mWeight["vcpu"]<<"\n";
		// std::cout << "RAM: "<< host->getResource()->mWeight["memory"] <<"\n";

		allocated_task[container->getId()]=(char*)host->getName().c_str();
		// std::cout << "Allocated! ID " << container->getId() << " HOST:"<<allocated_task[container->getId()]<<"!!!!!\n";
		// need to include the host name and container id in the allocated_task
		// std::cout<<"######################################\n";
		return true;
	}
	// std::cout<<"######################################\n";
	return false;
}

}
#endif
