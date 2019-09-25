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

bool links_allocator_cuda(Builder* builder,  Task* task, consumed_resource_t* consumed, int interval_low, int interval_high){
	spdlog::debug("Init Links Allocator");
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
	std::map<int,int> container_to_host;
	{
		std::map<int,int> pods_to_host;
		//get all the hosts in the topology that need to be calculated
		for(size_t i=0; i<containers_size; i++) {
			pods_to_host[containers[i]->getHostIdg()]=i;
		}
		hosts_size = pods_to_host.size();
		all_hosts_index = (int*) malloc (sizeof(int) * hosts_size );

		int i=0;
		for(std::map<int,int>::iterator it=pods_to_host.begin(); it!=pods_to_host.end(); it++) {
			all_hosts_index[i]=it->first;
			container_to_host[it->first]=i;
			spdlog::debug("-----------");
			spdlog::debug("{} -> {}",it->first,i);
			spdlog::debug("-----------");
			i++;
		}
		// printf("Total different hosts that need to calculate path %d\n", hosts_size);
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
	float *result = (float*) malloc (sizeof(float)*bytes_matrix);
	bool result_min_max[links_size];

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

	/*Free the host values*/
	free(all_hosts_index);

	spdlog::debug("Prebuilt Kernel check Erros {}\nLinks Allocator  kernel start",cudaGetErrorString(cudaGetLastError()));
	// spdlog::debug("Graph State {} - num_nodes {}", graph->get_state(), graph->get_num_nodes());
	{
		dim3 block(block_size,block_size,1);
		dim3 grid(ceil(hosts_size/(float)block.x), ceil(hosts_size/(float)block.y),1);
		/*Cuda Kernel Call*/
		widestPathKernel<<<grid,block>>>(
			graph->get_source_offsets(),
			graph->get_destination_indices(),
			graph->get_all_edges_ids(),
			graph->get_all_variable_edges(),
			graph->get_all_node_type(),
			d_path,
			d_path_edge,
			d_visited,
			d_weights,
			d_discount,
			d_all_hosts_index,
			d_result,
			hosts_size,
			nodes_size);
		cudaDeviceSynchronize();
		// exit(0);
		// getchar();
	}
	cudaError_t cuda_error = cudaGetLastError();
	if(cuda_error != cudaSuccess) {
		SPDLOG_ERROR("Links kernel error");
		exit(0);
	}
	if(cuda_error!=0) {
		SPDLOG_ERROR("Links Allocator - Kernel check Erros[x] !!!{}!!! Host Size: {} Node Size: {} Total: {}",cudaGetErrorString(cuda_error),hosts_size,nodes_size,((hosts_size*(hosts_size-1))/2)*nodes_size);
		exit(0);
	}

	/*Get the path values from to device and copy to host*/
	checkCuda( cudaMemcpy(path, d_path, bytes_matrix*  sizeof(int), cudaMemcpyDeviceToHost));
	checkCuda( cudaMemcpy(path_edge, d_path_edge, bytes_matrix*  sizeof(int), cudaMemcpyDeviceToHost));
	checkCuda( cudaMemcpy(result, d_result, bytes_matrix* sizeof(float), cudaMemcpyDeviceToHost));

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
	int walk_index=0, same_host=0;
	//Walk through all the containers
	//printf("Starting iterate the containers\n");
	spdlog::debug("\n\nInitializing the graph update, container_size {}",containers_size);
	for(size_t container_index=0; container_index<containers_size; container_index++) {
		//For each container get their links
		spdlog::debug("Running through container {} -> {}",container_index,containers[container_index]->getLinksSize());
		for(size_t i=0; i<containers[container_index]->getLinksSize(); i++) {
			links=containers[container_index]->getLinks();

			//walk through all the links of the specified container
			//check if the link between the container and the destination is in the same host, if is, do nothing.
			if(containers[links[i].destination-1]->getHostId() == containers[container_index]->getHostId()) {
				destination[link_index]=-1; //no need to calculate
				link_index++;
				same_host++;
				continue;
			}

			int initial_index=0;

			{
				int temp_src = container_to_host[containers[container_index]->getHostIdg()];
				int temp_dst = container_to_host[containers[links[i].destination-1]->getHostIdg()];
				if(temp_src < temp_dst) {
					initial_index=nodes_size*( temp_src*hosts_size+ temp_dst -discount[temp_src]);
				}else if(temp_src > temp_dst) {
					initial_index=nodes_size*( temp_dst*hosts_size+ temp_src -discount[temp_dst]);
				}else{
					destination[link_index]=-1; //no need to calculate
					link_index++;
					same_host++;
					continue;
				}

				// printf("!!%d!!\n", initial_index);

				spdlog::debug("CI {} | CSRC {} | SRC {} | I {} | CD {} | CDST {}\n",container_index, containers[container_index]->getHostIdg(), temp_src, i, containers[links[i].destination-1]->getHostIdg(), temp_dst);
			}
			// int initial_index = nodes_size*(container_index* hosts_size + (container_index+i+1-same_host) - discount[container_index]);


			spdlog::debug("Running through link {}",i);

			//If the containers are in different hosts, calculate the widestPath between them.
			//printf("Update the destination index\n");
			destination[link_index] = containers[links[i].destination-1]->getHostIdg();

			//Need to make the path and check if the path can support the bandwidth
			if(result[initial_index]>=links[i].bandwidth_max) {
				spdlog::debug("Link set with max band of {} the path has {}",links[i].bandwidth_max, result[initial_index]);
				result_min_max[link_index]=true;
				result[initial_index]-=links[i].bandwidth_max;
			}else if(result[initial_index]>=links[i].bandwidth_min) {
				spdlog::debug("Link set with min band of {} the path has {}",links[i].bandwidth_max, result[initial_index]);
				result_min_max[link_index]=false;
				result[initial_index]=links[i].bandwidth_min;
			}else{
				spdlog::debug("Link dont set, free the allocated links");
				freeHostResource(task, builder, interval_low, interval_high);
				freeLinks(task, consumed, builder, link_index);
				spdlog::debug("Links removed\n");
				spdlog::info("Link has less resources than pod asked");
				return false;
			}

			//allocate the link
			values[link_index] = (result_min_max[link_index]) ? links[i].bandwidth_max : links[i].bandwidth_min;

			walk_index = destination[link_index];
			//printf("Walink in the path and reducing the value inside the topology\n");

			init[link_index]=initial_index;

			// spdlog::error("Path[{}]={}",initial_index+walk_index,path[initial_index+walk_index]);
			while(path[initial_index+walk_index]!=-1) {
				spdlog::debug("Updating the edge {}",initial_index+walk_index);
				graph->set_variable_edge(
					1,      //to refer to the bandwidth
					path_edge[initial_index+walk_index],      //to refer to the specific edge
					(graph->get_variable_edge( 1, path_edge[initial_index+walk_index ] ) - values[link_index])
					);

				consumed->total_bandwidth_consumed += values[link_index];

				graph->add_connection_edge( path_edge[initial_index + walk_index] );

				walk_index = path[initial_index+walk_index];
			}
			spdlog::debug("Add link index");
			//update the link_index
			link_index++;
		}
	}
	consumed->active_links = graph->get_num_active_edges();
	spdlog::debug("End Links Allocator with 0");
	return true;
}

}

#endif
