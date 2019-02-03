
#ifndef _WORST_FIT_NOT_INCLUDED_
#define _WORST_FIT_NOT_INCLUDED_

#include <iostream>
#include <string>
#include <map>

#include "../../datacenter/tasks/container.hpp"
#include "../../builder.cuh"
#include "../utils.hpp"

#include <limits.h>
#include <queue>

//OPCOES
//    //COLOCAR ESSA FUNÇÃO EM GPU, REAPROVEITAR A FUNÇÃO MAX QUE EXISTE NO TOPSIS
//    //COLOCAR EM UM PRIORITY QUEUE E RETIRAR O PRIMEIRO ELEMENTO
// PARA AMBAS AS ABORDAGENS, PRECISA ACHAR UM MEIO DE JUNTAR OS RECURSOS DO CONTAINER.
namespace Allocator {
bool WorstFit(Builder* builder,  Container* container, std::map<unsigned int, unsigned int> &allocated_task,consumed_resource_t* consumed){

	std::priority_queue<Host*> hosts (builder->getHosts().begin(), builder->getHosts().end());

	int fit=checkFit(hosts.top(),container);
	if(fit==0) {
		return false;
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

	allocated_task[container->getId()]= host->getId();

	return true;
}
}
#endif
