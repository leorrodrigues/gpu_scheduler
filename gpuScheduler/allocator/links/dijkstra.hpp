#ifndef _DIJSKTRA_NOT_DEFINED_
#define _DIJKSTRA_NOT_DEFINED_

#include <algorithm>
#include <cfloat>
#include <queue>
#include <list>

#include "../builder.cuh"
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
int* dijkstra(vnegpu::graph<float>*dc, int src, int dst, float b_max, float b_min, bool& res){
	res=true; //allocating b_max

	int *nodes_types = dc->get_all_node_type();
	unsigned int nodes_size = dc->get_num_nodes();

	//Priority queue to store all the (distance,vertex)
	std::priority_queue<
		std::pair<float,int>,
		std::vector <std::pair<float,int> >
		> pq;
	float weights[nodes_size];

	int *path = (int*) malloc(sizeof(int)*nodes_size);

	bool visited[nodes_size];

	for(size_t i=0; i<nodes_size; i++) {
		weights[i]=FLT_MIN;
		path[i]=-1;
		visited[i]= (nodes_types[i]==0) ? true : false;
	}
	visited[src]=false;
	visited[dst]=false;

	weights[src]=0;
	path[src]=src;
	pq.push_back(std::make_pair(0,src));

	unsigned int node_index, next_node;
	size_t destination_index;
	float next_node_weight;
	while(!pq.empty()) {
		node_index=pq.top().second;
		pq.pop();

		if(!visited[node_index]) {
			visited[node_index]=true;

			//need to run through all the edges of the Node_index
			for(destination_index= dc->get_source_offset(node_index);
			    destination_index< dc->get_source_offset(node_index+1);
			    destination_index++) {

				next_node = dc->get_destination_indice(destination_index);
				next_node_weight = dc->get_variable_edge ( 1, dc->get_edges_ids(next_node));


				if( (!visited[next_node]) && (weights[next_node] < (weights[node_index] + next_node_weight))) {
					if(next_node_weight < b_max && next_node_weight > b_min) {
						res=false; //allocating b_min
					}else if(next_node_weight < b_min) {
						free(path);
						return NULL; //can't allocate
					}

					weights[next_node] = weights[node_index] + next_node_weight;
					path[next_node]=node_index;
					pq.push(std::make_pair(weights[next_node],next_node));

				}
			}
		}
	}
	return path;
}

}

#endif
