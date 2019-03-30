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
	int devID;
	cudaDeviceProp props;
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&props, devID);
	int block_size = (props.major <2) ? 16 : 32;

	const unsigned short discount_size = 48;
	int discount[] = {1,3,6,10,15,21,28,36,45,55,66,78,91,105,120,136,153,171,190,210,231,253,276,300,325,351,378,406,435,465,496,528,561,595,630,666,703,741,780,820,861,903,946,990,1035,1081,1128,1176}; //used inside the kernel only;

	/*Task Variables*/
	Container** containers = task->getContainers();
	unsigned int containers_size = task->getContainersSize();
	unsigned int links_size = task->getLinksSize();

	/*Topology Variables*/
	vnegpu::graph<float>* graph = builder->getTopology()->getGraph();
	const unsigned int nodes_size = graph->get_num_nodes();
	size_t hosts_size;

	/*Get all the hosts graph index*/
	int *all_hosts_index;
	{
		std::map<int,int> pods_to_host;
		//get all the hosts in the topology that need to be calculated
		for(size_t i=0; i<containers_size; i++) {
			pods_to_host[i]=containers[i]->getHostIdg();
		}
		hosts_size = pods_to_host.size();
		all_hosts_index = (int*) malloc (sizeof(int) * hosts_size );

		for(std::map<int,int>::iterator it=pods_to_host.begin(); it!=pods_to_host.end(); it++) {
			all_hosts_index[it->first]=it->second;
		}
	}

	size_t bytes_matrix = ((hosts_size*(hosts_size-1))/2)*nodes_size;

	//Initialize host variables
	float *values      = (float*) malloc (links_size   *sizeof(float));
	int   *path        = (int*)   malloc (bytes_matrix *  sizeof(int));
	int   *path_edge   = (int*)   malloc (bytes_matrix *  sizeof(int));
	int   *destination = (int*)   malloc (links_size   *  sizeof(int));
	int   *init        = (int*)   malloc (links_size   *  sizeof(int));

	//Update the variables inside the task
	task->setLinkPath(path);
	task->setLinkPathEdge(path_edge);
	task->setLinkDestination(destination);
	task->setLinkInit(init);
	task->setLinkValues(values);

	/*Result BW */
	float *result = (float*) malloc (sizeof(float)*(hosts_size*(hosts_size-1))/2);

	//Create the device variables
	int   *d_discount, *d_all_hosts_index, *d_path, *d_path_edge;
	float *d_weights, *d_result;
	bool  *d_visited;

	/*Malloc the device memory*/
	checkCuda( cudaMalloc((void**)&d_discount,    discount_size* sizeof(int)));
	checkCuda( cudaMalloc((void**)&d_all_hosts_index, hosts_size*sizeof(int)));
	checkCuda( cudaMalloc((void**)&d_path,        bytes_matrix*  sizeof(int)));
	checkCuda( cudaMalloc((void**)&d_path_edge,   bytes_matrix*  sizeof(int)));
	checkCuda( cudaMalloc((void**)&d_result,      bytes_matrix*sizeof(float)));
	checkCuda( cudaMalloc((void**)&d_weights,     bytes_matrix*sizeof(float)));
	checkCuda( cudaMalloc((void**)&d_visited,     bytes_matrix* sizeof(bool)));

	/*Set the values inside each variable*/
	checkCuda( cudaMemcpy(d_discount, discount, discount_size*sizeof(int), cudaMemcpyHostToDevice));
	checkCuda( cudaMemcpy(d_all_hosts_index, all_hosts_index, hosts_size*sizeof(int), cudaMemcpyHostToDevice));
	checkCuda( cudaMemset(d_path,        -1, bytes_matrix*  sizeof(int)));
	checkCuda( cudaMemset(d_path_edge,   -1, bytes_matrix*  sizeof(int)));
	checkCuda( cudaMemset(d_result,       0, bytes_matrix*sizeof(float)));
	checkCuda( cudaMemset(d_weights,FLT_MIN, bytes_matrix*sizeof(float)));

	checkCuda( cudaMemset(d_visited,  false, ((hosts_size*(hosts_size-1))/2)* sizeof(bool)));

	/*Free the host values*/
	free(all_hosts_index);

	{
		dim3 block(block_size,block_size,1);
		dim3 grid(ceil(hosts_size/(float)block.x), ceil(hosts_size/(float)block.y),1);
		/*Cuda Kernel Call*/
		widestPathKernel<<<grid,block>>>(
			graph,
			d_path,
			d_path_edge,
			hosts_size,
			nodes_size,
			d_visited,
			d_weights,
			d_discount,
			d_all_hosts_index,
			d_result
		                );
		cudaDeviceSynchronize();
	}

	/*Get the path values from to device and copy to host*/
	checkCuda( cudaMemcpy(path, d_path, bytes_matrix*  sizeof(int), cudaMemcpyDeviceToHost));
	checkCuda( cudaMemcpy(path_edge, d_path_edge, bytes_matrix*  sizeof(int), cudaMemcpyDeviceToHost));
	checkCuda( cudaMemcpy(result, d_result, ((hosts_size*(hosts_size-1))/2)* sizeof(float), cudaMemcpyDeviceToHost));

	/*Free all the device variables*/
	checkCuda( cudaFree(d_path));
	checkCuda( cudaFree(d_result));
	checkCuda( cudaFree(d_weights));
	checkCuda( cudaFree(d_visited));
	checkCuda( cudaFree(d_discount));
	checkCuda( cudaFree(d_path_edge));
	checkCuda( cudaFree(d_all_hosts_index));

	//Check if all paths can be constructed
	//Now we need to construct the path and update the graph values
	unsigned int link_index = 0;
	Link* links = NULL;
	int walk_index=0;
	//Walk through all the containers
	//printf("Starting iterate the containers\n");

	for(size_t container_index=0; container_index<containers_size; container_index++) {
		//For each container get their links
		for(size_t i=0; i<containers[container_index]->getLinksSize(); i++) {
			links=containers[container_index]->getLinks();
			//walk through all the links of the specified container
			//check if the link between the container and the destination is in the same host, if is, do nothing.
			if(containers[links[i].destination-1]->getHostId() == containers[container_index]->getHostId()) {
				destination[link_index]=-1;      //no need to calculate
				link_index++;
				continue;
			}
			//If the containers are in different hosts, calculate the widestPath between them.
			//printf("Update the destination index\n");
			destination[link_index] = containers[links[i].destination-1]->getHostIdg();

			//Need to make the path and check if the path can support the bandwidth
			if(result[link_index]>=links[i].bandwidth_max) {
				result[link_index]=2;
			}else if(result[link_index]>=links[i].bandwidth_min) {
				result[link_index]=1;
			}else{
				result[link_index]=0;
			}

			//Now we have the shortest path between src and the destination. If the link can't be set between the [SRC, DST], the result value in the result[index] == 0, if the minimum can be allocated, the value is 1, if maximum, the value is 2.
			if(result[link_index]==0) {
				//free all the resources allocated
				freeHostResource(task,consumed,builder);
				//free the link allocated
				freeLinks(task,consumed,builder, link_index);

				return false;
			}
			//allocate the link
			values[link_index] = (result[link_index]==2) ? links[i].bandwidth_max : links[i].bandwidth_min;

			walk_index = destination[link_index];
			// //printf("Walink in the path and reducing the value inside the topology\n");
			int initial_index = nodes_size*(container_index* hosts_size + i - discount[container_index]);
			init[link_index]=initial_index;

			while(path[initial_index+walk_index]!=-1) {
				graph->set_variable_edge(
					1,      //to refer to the bandwidth
					path_edge[initial_index+walk_index],      //to refer to the specific edge
					(graph->get_variable_edge( 1, path_edge[initial_index+walk_index ] ) - values[link_index])
					);

				consumed->resource["bandwidth"] += values[link_index];

				graph->add_connection_edge( path_edge[initial_index + walk_index] );

				walk_index = path[initial_index+walk_index];
			}
			//update the link_index
			link_index++;
		}
	}
	consumed->active_links = graph->get_num_active_edges();
	return true;
}

}

#endif
