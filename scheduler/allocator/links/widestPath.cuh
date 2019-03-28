#ifndef _WIDEST_PATH_CUDA_NOT_DEFINED_
#define _WIDEST_PATH_CUDA_NOT_DEFINED_

#include <cfloat>

#include "../../builder.cuh"
#include "../../thirdparty/vnegpu/graph.cuh"

namespace Allocator {
/**
   Dijkstra function to find the shortest path between the all the hosts in the Data Center. This implementation aims to find the path with the bigger value.
   DC is the graph topology
   Path is a 3D matrix containing all paths between all nodes.
   Path_edge is a 3D matrix containing all the edges that connect the nodes described in Path
   BW is the array containing min and max bandwidth that the container requested
   res is the result variable, used as a manager variable to tell if the found path can support the maximum or minimum bandwidth.
   Nodes_Types is the array obtained through the Topology Graph, indicating if a node is host, switch or switch core.
   Visited is the array indicating if the speific node has been already visited.
   Weights is the simulation of Priority queue used to guide the widest path, all values that has to be ignored as value 0.
   Size is the total of host nodes in graph.
   Nodes_size is the total number of nodes in graph.
 */
__device__
void widestPathKernel(vnegpu::graph<float>*dc, int* path, int* path_edge, int* nodes_types, const unsigned int host_size, const unsigned int nodes_size, bool *visited, float *weights, int *discount){
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	if(x >= host_size-1 || y >= host_size-1 || x>=y) return;
	//Need to find de source and the destination in the graph through the X and Y variables... HOW?
	const int src = ?;
	const int dst = ?;
	/*
	    Exemple: 6 hosts -> The array is composed by :
	    0-01 1-02 2-03 3-04 4-05
	    5-12 6-13 7-14 8-15
	    9-23 10-24 11-25
	    12-34 13-35
	    14-45

	   The matrix has [(|host_size|*(|host_size|-1))/2]*nodes_size
	 */

	int initial_index = nodes_size*(x* host_size + y   - discount[x]);

	weights[initial_index+src]=FLT_MAX;
	visited[initial_index+src]=false;
	visited[initial_index+dst]=false;

	int node_index, next_node;
	size_t destination_index;
	float next_node_weight, alt;
	cublasHandle_t cublasHandle;
	while(true) {
		cublasIsamax(cublasHandle, nodes_size, weights, 1,&node_index);

		if(weights[initial_index+node_index]==FLT_MIN || node_index==dst) break; //simulate empty queue or we found the destination node

		//need to run through all the edges of the Node_index
		for(destination_index= dc->get_source_offset(node_index);
		    destination_index< dc->get_source_offset(node_index+1);
		    destination_index++) {

			next_node = dc->get_destination_indice(destination_index);

			if(visited[initial_index+next_node]) continue; //if the node is another host, ignore it.

			next_node_weight = dc->get_variable_edge ( 1, dc->get_edges_ids(destination_index)); // get the value of the edge bettween U and V.

			alt = weights[initial_index+node_index] < next_node_weight ?
			      (
				weights[initial_index+next_node] > weights[initial_index+node_index] ?
				weights[initial_index+next_node] : weights[initial_index+node_index]
			      )
			      :
			      (
				weights[initial_index+next_node] > next_node_weight ?
				weights[initial_index+next_node] : next_node_weight
			      );


			if(alt> weights[initial_index+next_node]) {
				weights[initial_index+next_node] = alt;
				path[initial_index+next_node] = node_index;
				path_edge[initial_index+next_node] = dc->get_edges_ids(destination_index);
				weights[initial_index+next_node]=next_node;
			}
		}
	}
}

}

#endif