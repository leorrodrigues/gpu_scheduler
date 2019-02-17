#ifndef _LINKS_ALLOCATOR_NOT_DEFINED_
#define _LINKS_ALLOCATOR__NOT_DEFINED_

#include <algorithm>
#include <cfloat>
#include <queue>
#include <list>

#include "../builder.cuh"
#include "dijkstra.hpp"

!!!PRECISA COLOCAR UM PATHS**QUE CONTEM TODOS OS PATH DO DIJKSTRA, POIS CASO UM LINK NAO DE CERTO, PRECISA DESFAZER TODAS AS ALOCACOES, DESTE MODO, TENDO O PATHS** TU CONSEGUE FAZER ISSO.

namespace Allocator {

bool links_allocator(Builder* builder,  Task* task, consumed_resource_t* consumed){
	Container** containers = task->getContainers();
	Link* links=NULL;
	int *path=NULL;
	Topology* topology;

	bool max_min=false; //if true max is set, min otherwise
	unsigned int containers_size = task->getContainersSize();
	unsigned int links_size=0;
	float value=0;
	size_t index,i,j;

	for(i=0; i<containers_size; i++) {
		links_size = containers[i]->getLinksSize();
		if(links_size>0) {
			links=containers[i]->getLinks();
			for(j=0; j<links_size; j++) {
				//check if the destination container is in the same host than the source.
				if(containers[links[i].destination]->getHostId() == containers[i]->getHostId())
					continue;

				path = dijkstra(
					topology->getGraph(),
					i,
					links[i].destination,
					links[i].bandwidth_max,
					links[i].bandwidth_min,
					max_min);

				//Now we have the shortest path between src and the destination, all the other hosts has their values in the path set as -1. The switchs may have different values.
				//Check if we can allocate the link, if path==NULL the link cannot be set.
				if(path==NULL) {
					//need to undo all the pod's allocation

					//need to undo all the link's allocation

					return false;
				}
				//allocate the link
				value = (max_min==true) ? links[i].bandwidth_max : links[i].bandwidth_min;
				//Start the allocation through the destination
				index = links[i].destination;
				while(index!=i) {
					//subtract the value

					//update the index
					index=path[index];
				}
			}
		}
	}
	return true;
}

}

#endif
