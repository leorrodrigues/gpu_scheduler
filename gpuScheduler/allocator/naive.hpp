#include <iostream>
#include <string>
#include <map>

#include "../datacenter/tasks/container.hpp"
#include "../builder.cuh"

namespace Allocator {

inline bool checkFit(Host* host, Container* container){
	if(host->getResource()->mWeight["vcpu"]<container->containerResources->vcpu_max) {
		std::cout<<"VCPU "<<host->getResource()->mWeight["vcpu"]<<" AND "<<container->containerResources->vcpu_max;
		return false;
	}
	if(host->getResource()->mWeight["memory"]<container->containerResources->ram_max) {
		std::cout<<"Memory "<<host->getResource()->mWeight["memory"]<<" AND "<<container->containerResources->ram_max<<"\n";
		return false;
	}
	return true;
}

bool naive(Builder* builder,  Container* container, std::map<int,std::string> &allocated_task){
	std::cout << "##############################\nTry the allocation\n";

	std::cout << "Running Multicriteria\n";
	// builder->listHosts();

	builder->runMulticriteria( builder->getHosts() );

	std::cout << "Multicriteria OK\n";
	std::cout << "Getting Results\n";
	std::map<int,std::string> result = builder->getMulticriteriaResult();

	// Para testes
	std::cout << "Results Found!\nAllocating\n";
	Host* host=NULL;
	for( std::map<int,std::string>::iterator it = result.begin(); it!=result.end(); it++) {
		std::cout<< "Checking the host "<<it->first<<":"<<it->second<<"\n";
		host=builder->getHost(it->second);
		std::cout << "HOST RESOURCES\n";
		std::cout << "VCPU: "<< host->getResource()->mWeight["vcpu"]<<"\n";
		std::cout << "RAM: "<< host->getResource()->mWeight["memory"] <<"\n";

		if(!checkFit(host,container)) {
			std::cout<<"This host cant support the container\n";

			continue;
		}

		std::cout<<"Subtracting the resources\n";
		(*host)-=(*container);
		std::cout<<"Subtracted\n";
		std::cout << "HOST RESOURCES 2\n";
		std::cout << "VCPU: "<< host->getResource()->mWeight["vcpu"]<<"\n";
		std::cout << "RAM: "<< host->getResource()->mWeight["memory"] <<"\n";

		allocated_task[container->getId()]=host->getName();
		std::cout << "Allocated! ID " << container->getId() << " HOST:"<<allocated_task[container->getId()]<<"!!!!!\n";
		// need to include the host name and container id in the allocated_task
		std::cout<<"######################################\n";
		return true;
	}
	std::cout<<"######################################\n";
	return false;
}

}
