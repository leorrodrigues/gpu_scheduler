#ifndef _LINKS_ALLOCATOR_CUDA_NOT_DEFINED_
#define _LINKS_ALLOCATOR_CUDA_NOT_DEFINED_

#include <cfloat>

#include "../free.hpp"

#include "../../datacenter/tasks/task.hpp"
#include "../../datacenter/tasks/container.hpp"
#include "../../builder.cuh"
#include "widestPath.cuh"

inline
cudaError_t checkCuda(cudaError_t result){
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		SPDLOG_ERROR("CUDA Runtime Error: {}", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

namespace Allocator {

bool links_allocator_cuda(Builder* builder,  Task* task, consumed_resource_t* consumed){
	const unsigned short discount_size = 48;
	int discount[] = {1,3,6,10,15,21,28,36,45,55,66,78,91,105,120,136,153,171,190,210,231,253,276,300,325,351,378,406,435,465,496,528,561,595,630,666,703,741,780,820,861,903,946,990,1035,1081,1128,1176}; //used inside the kernel only;

	Container** containers = task->getContainers();
	unsigned int containers_size = task->getContainersSize();
	unsigned int links_size = task->getLinksSize();

	vnegpu::graph<float>* graph = builder->getTopology()->getGraph();
	const unsigned int nodes_size = graph->get_num_nodes();
	const unsigned int hosts_size = builder->getHostsSize();
	//Initialize host variables
	float *values = (float*) malloc (sizeof(float)*links_size);
	int **path = (int**) malloc(sizeof(int*)*links_size);
	int **path_edge = (int**)malloc(sizeof(int*)*links_size);
	for(size_t i=0; i<links_size; i++) {
		path[i] = (int*) malloc(sizeof(int)*nodes_size);
		path_edge[i] = (int*)malloc(sizeof(int)*nodes_size);
	}
	int *destination = (int*) malloc(sizeof(int)*links_size);

	//Update the variables inside the task
	task->setLinkPath(path);
	task->setLinkPathEdge(path_edge);
	task->setLinkDestination(destination);
	task->setLinkValues(values);

	//Create the device variables
	int   *d_discount, *d_path, *d_path_edge;
	float *d_weights;
	bool  *d_visited;

	size_t bytes_matrix = ((hosts_size*(hosts_size-1))/2)*nodes_size;

	/*Malloc the device memory*/
	checkCuda( cudaMalloc((void**)&d_discount, discount_size* sizeof(int)));
	checkCuda( cudaMalloc((void**)&d_path,     bytes_matrix*  sizeof(int)));
	checkCuda( cudaMalloc((void**)&d_path_edge,bytes_matrix*  sizeof(int)));
	checkCuda( cudaMalloc((void**)&d_weights,  bytes_matrix*sizeof(float)));
	checkCuda( cudaMalloc((void**)&d_visited,  bytes_matrix* sizeof(bool)));

	/*Set the values inside each variable*/
	checkCuda( cudaMemcpy(d_discount, discount, discount_size*sizeof(int), cudaMemcpyHostToDevice));
	checkCuda( cudaMemset(d_path,        -1, bytes_matrix*  sizeof(int)));
	checkCuda( cudaMemset(d_path_edge,   -1, bytes_matrix*  sizeof(int)));
	checkCuda( cudaMemset(d_weights,FLT_MIN, bytes_matrix*sizeof(float)));
	checkCuda( cudaMemset(d_visited,  false, bytes_matrix* sizeof(bool)));

	widestPathKernel(
		graph, //the graph to calculate
		d_path,
		d_path_edge,
		graph->get_all_node_type(),
		// containers[container_index]->getHostIdg(), // the source host
		// containers[links[i].destination-1]->getHostIdg(), // the destination host
		hosts_size,
		nodes_size,
		d_visited,
		d_weights,
		d_discount
		);

	// unsigned int result[links_size];
	// unsigned int link_index = 0;
	// Link* links = NULL;
	// int walk_index=0;
	// //Walk through all the containers
	// //printf("Starting iterate the containers\n");
	//
	// for(size_t container_index=0; container_index<containers_size; container_index++) {
	//      //printf("Looking the container %d\n",container_index);
	//      //For each container get their links
	//      //printf("This container has %d links\n", containers[container_index]->getLinksSize());
	//      for(size_t i=0; i<containers[container_index]->getLinksSize(); i++) {
	//              //printf("Looking for link %d\n",i);
	//              links=containers[container_index]->getLinks();
	//              //walk through all the links of the specified container
	//              //check if the link between the container and the destination is in the same host, if is, do nothing.
	//              //printf("Cheking if the link is between two containers that are in the same host\n");
	//              //printf("Containers ci[%d] = %d | graph %d\n",container_index,containers[container_index]->getHostId(), containers[container_index]->getHostIdg());
	//              //printf("Destination %d\n", links[i].destination),
	//              //printf("Containers[destination]=%d| graph %d\n",containers[links[i].destination-1]->getHostId(), containers[links[i].destination-1]->getHostIdg());
	//              if(containers[links[i].destination-1]->getHostId() == containers[container_index]->getHostId()) {
	//                      //printf("\t\t\t!!!!!!!!!The containers are in the same host\n");
	//                      destination[link_index]=-1;         //no need to calculate
	//                      link_index++;
	//                      continue;
	//              }
	//              //If the containers are in different hosts, calculate the widestPath between them.
	//              //printf("Update the destination index\n");
	//              destination[link_index] = containers[links[i].destination-1]->getHostIdg();
	//
	//              // printf("Call the widestPath\n");
	//
	//              // printf("Widest path calculated\n");
	//
	//              //Now we have the shortest path between src and the destination. If the link can't be set between the [SRC, DST], the result value in the result[index] == 0, if the minimum can be allocated, the value is 1, if maximum, the value is 2.
	//              if(result[link_index]==0) {
	//                      // //printf("Desalocating\n");
	//                      //free all the resources allocated
	//                      freeHostResource(task,consumed,builder);
	//                      //free the link allocated
	//                      freeLinks(task,consumed,builder, link_index);
	//                      // //printf("Desalocated\n");
	//                      return false;
	//              }
	//              // //printf("Allocating the link\n");
	//              //allocate the link
	//              values[link_index] = (result[link_index]==2) ? links[i].bandwidth_max : links[i].bandwidth_min;
	//
	//              walk_index = destination[link_index];
	//              // //printf("Walink in the path and reducing the value inside the topology\n");
	//              while(path[link_index][walk_index]!=-1) {
	//                      graph->set_variable_edge(
	//                              1,         //to refer to the bandwidth
	//                              path_edge[link_index][walk_index],         //to refer to the specific edge
	//                              (graph->get_variable_edge( 1, path_edge[link_index][ walk_index ] ) - values[link_index])
	//                              );
	//
	//                      consumed->resource["bandwidth"] += values[link_index];
	//
	//                      graph->add_connection_edge(path_edge[link_index][walk_index]);
	//
	//                      walk_index = path[link_index][walk_index];
	//              }
	//              //printf("---------------------------------\n\n\n");
	//              //update the link_index
	//              link_index++;
	//      }
	// }
	// consumed->active_links = graph->get_num_active_edges();
	return true;
}

}

#endif
