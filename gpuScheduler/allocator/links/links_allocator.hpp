#ifndef _LINKS_ALLOCATOR_NOT_DEFINED_
#define _LINKS_ALLOCATOR__NOT_DEFINED_

#include <algorithm>
#include <cfloat>
#include <queue>
#include <list>

#include "../../datacenter/tasks/container.hpp"
#include "../../builder.cuh"
#include "widestPath.hpp"

namespace Allocator {

bool links_allocator(Builder* builder,  Task* task, consumed_resource_t* consumed){
	Container** containers = task->getContainers();
	unsigned int containers_size = task->getContainersSize();
	unsigned int links_size = task->getLinksSize();

	vnegpu::graph<float>* graph = builder->getTopology()->getGraph();
	unsigned int nodes_size = graph->get_num_nodes();

	int path[links_size][nodes_size];
	int path_edge[links_size][nodes_size];
	unsigned int result[links_size]; //0 for ERROR, 1 for MIN, 2 for MAX
	int destination [links_size];
	unsigned int link_index = 0;
	Link* links = NULL;

	float values[link_index];
	int walk_index=0;
	//Walk through all the containers
	printf("Starting iterate the containers\n");
	for(size_t container_index=0; container_index<containers_size; container_index++) {
		printf("Looking the container %d\n",container_index);
		//For each container get their links
		printf("This container has %d links\n", containers[container_index]->getLinksSize());
		for(size_t i=0; i<containers[container_index]->getLinksSize(); i++) {
			printf("Looking for link %d\n",i);
			links=containers[container_index]->getLinks();
			//walk through all the links of the specified container
			//check if the link between the container and the destination is in the same host, if is, do nothing.
			printf("Cheking if the link is between two containers that are in the same host\n");
			printf("Containers ci[%d] = %d\n",container_index,containers[container_index]->getHostId());
			printf("Destination %d\n", links[i].destination),
			printf("Containers[destination]=%d\n",containers[links[i].destination-1]->getHostId());
			if(containers[links[i].destination-1]->getHostId() == containers[container_index]->getHostId()) {
				destination[link_index]=-1;         //no need to calculate
				link_index++;
				continue;
			}
			//If the containers are in different hosts, calculate the widestPath between them.
			printf("Update the destination index\n");
			destination[link_index] = containers[links[i].destination-1]->getHostIdg();

			printf("Call the widestPath\n");
			result[link_index]= widestPath(
				graph,         //the graph to calculate
				containers[container_index]->getHostIdg(),         // the source host
				containers[links[i].destination-1]->getHostIdg(),         // the destination host
				path[link_index],
				path_edge[link_index],
				links[i].bandwidth_min,
				links[i].bandwidth_max
				);
			printf("Widest path calculated\n");

			//Now we have the shortest path between src and the destination. If the link can't be set between the [SRC, DST], the result value in the result[index] == 0, if the minimum can be allocated, the value is 1, if maximum, the value is 2.
			if(result[link_index]==0) {
				//need to undo all the pod's allocation
				Pod** pods =  task->getPods();
				for(size_t p = 0; p<task->getPodsSize(); p++) {
					freeHostResource(pods[p],consumed,builder);
				}
				//need to undo all the link's allocation
				for(size_t p = 0; p<link_index; p++) {

					walk_index = destination[p];
					if(walk_index == -1 ) continue;         // the link is between two containers that are in the same host

					while(path[p][walk_index]!=-1) {
						graph->set_variable_edge_undirected(
							1,         //to refer to the bandwidth
							path_edge[p][walk_index],         //to refer to the specific edge
							(graph->get_variable_edge( 1, path_edge[p][ walk_index ] ) - values[p])
							);
						walk_index = path[p][walk_index];
					}
				}
				return false;
			}
			//allocate the link
			values[link_index] = (result[link_index]==2) ? links[i].bandwidth_max : links[i].bandwidth_min;

			walk_index = destination[link_index];
			while(path[link_index][walk_index]!=-1) {
				graph->set_variable_edge_undirected(
					1,         //to refer to the bandwidth
					path_edge[link_index][walk_index],         //to refer to the specific edge
					(graph->get_variable_edge( 1, path_edge[link_index][ walk_index ] ) - values[link_index])
					);
				walk_index = path[link_index][walk_index];
			}

			//update the link_index
			link_index++;
		}
	}
	return true;
}

}

#endif
