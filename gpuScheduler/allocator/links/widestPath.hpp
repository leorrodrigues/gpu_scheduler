#ifndef  _WIDEST_PATH_NOT_DEFINED_
#define _WIDEST_PATH_NOT_DEFINED_

#include <algorithm>
#include <cfloat>
#include <queue>

#include "../../builder.cuh"
#include "../../thirdparty/vnegpu/graph.cuh"

namespace Allocator {
/**
   Dijkstra function to find the shortest path between the two hosts in the Data Center. This implementation aims to find the path with the bigger value.
   SRC is the index of the source host in the DC.
   DST is the index of the destination host in the DC.
   b_max is the max bandwidth that the container requested
   b_min is the minimun bandwidth that the container requested
   res is the result variable, used as a manager variable to tell if the found path can support the maximum or minimum bandwidth.
 */
unsigned int widestPath(vnegpu::graph<float>*dc, int src, int dst, int* path, int* path_edge, float b_min, float b_max){
	// printf("SRC %d | DST %d |  TOTAL %d\n",src,dst, dc->get_num_nodes());
	int *nodes_types = dc->get_all_node_type();
	unsigned int nodes_size = dc->get_num_nodes();

	//Priority queue to store all the (distance,vertex)
	std::priority_queue<
		std::pair<float,int>,
		std::vector <std::pair<float,int> >
		> pq;

	bool visited[nodes_size];
	float weights[nodes_size];

	for(size_t i=0; i<nodes_size; i++) {
		weights[i]=FLT_MIN;
		path[i]=-1;
		path_edge[i]=-1;
		visited[i] = (nodes_types[i]==0) ? true : false;
	}

	weights[src]=FLT_MAX;
	visited[src]=false;
	visited[dst]=false;

	unsigned int node_index, next_node;
	size_t destination_index;
	float next_node_weight, alt;

	pq.push(std::make_pair(0,src));
	// printf("Start Iteration\n");
	while(!pq.empty()) {
		// printf("Getting the node index\n");
		node_index=pq.top().second;
		pq.pop();

		// printf("Cheking if i was in the limit\n");
		if(weights[node_index]==FLT_MIN || node_index==dst)
			break;

		//need to run through all the edges of the Node_index
		// printf("Iterating throught the edges of node_index %d\n",node_index);
		for(destination_index= dc->get_source_offset(node_index);
		    destination_index< dc->get_source_offset(node_index+1);
		    destination_index++) {

			// printf("Getting the next node destination\n");
			next_node = dc->get_destination_indice(destination_index);

			// printf("\tIf this node is a HOST ignore it\n");
			if(visited[next_node]) continue; //if the node is another host, ignore it.

			// printf("Get the weight of this node\n");
			next_node_weight = dc->get_variable_edge ( 1, dc->get_edges_ids(destination_index)); // get the value of the edge bettween U and V.

			// printf("Get the value\n");
			alt = std::max(weights[next_node], std::min(weights[node_index], next_node_weight));

			// printf("Check if its necessary to update\n");
			if(alt> weights[next_node]) {
				// printf("Updating\n");
				weights[next_node] = alt;
				path[next_node] = node_index;
				path_edge[next_node] = dc->get_edges_ids(destination_index);
				pq.push(std::make_pair(weights[next_node],next_node));
			}
			// printf("Iteration made\n");
			// printf("------------------------------------\n");
		}
	}

	//if didnt find the destination in de graph
	if(path[dst]==-1)
		return 0;

	int index=dst;
	float value = FLT_MAX;
	float temp=0;
	//wall through the path, starting from the destination to source.
	while(path[index]!=-1) {
		temp = dc->get_variable_edge(1, path_edge[index]);
		value = (value > temp) ? temp : value;
		index=path[index];
	}

	//check if we can allocate max, min bandwidth, 0 if cant.
	if(value> b_max)
		return 2;
	else if(value> b_min)
		return 1;
	else
		return 0;
}

}

#endif
