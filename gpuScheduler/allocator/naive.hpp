#ifndef _NAIVE_ALLOCATION_
#define _NAIVE_ALLOCATION_

#include <iostream>
#include <string>
#include <map>

#include "../datacenter/tasks/container.hpp"
#include "../builder.cuh"
#include "utils.hpp"

namespace Allocator {

bool naive(Builder* builder,  Container* container, std::map<int,const char*> &allocated_task,consumed_resource_t* consumed){
	// std::cout << "##############################\nTry the allocation\n";

	// std::cout << "Running Multicriteria\n";
	// builder->listHosts();

	builder->runMulticriteria( builder->getHosts() );

	// std::cout << "Multicriteria OK\n";
	// std::cout << "Getting Results\n";
	std::map<int,const char*> result = builder->getMulticriteriaResult();

	// Para testes
	// std::cout << "Results Found!\nAllocating\n";
	Host* host=NULL;
	for( std::map<int,const char*>::iterator it = result.begin(); it!=result.end(); it++) {
		// std::cout<< "Checking the host "<<it->first<<":"<<it->second<<"\n";
		host=builder->getHost(std::string(it->second));
		// std::cout << "VCPU: "<< host->getResource()->mWeight["vcpu"]<<"\n";
		// std::cout << "HOST RESOURCES\n";
		// std::cout << "RAM: "<< host->getResource()->mWeight["memory"] <<"\n";

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
		// std::cout<<"Subtracted\n";
		// std::cout << "HOST RESOURCES 2\n";
		// std::cout << "VCPU: "<< host->getResource()->mWeight["vcpu"]<<"\n";
		// std::cout << "RAM: "<< host->getResource()->mWeight["memory"] <<"\n";

		char* host_name = (char*) malloc (strlen(host->getName().c_str())+1);
		strcpy(host_name,host->getName().c_str());

		allocated_task[container->getId()]= &host_name[0];

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
