#ifndef _WIDEST_PATH_CUDA_NOT_DEFINED_
#define _WIDEST_PATH_CUDA_NOT_DEFINED_

#include <cfloat>

#include "../../builder.cuh"
#include "../../thirdparty/vnegpu/graph.cuh"


namespace Allocator {

__device__
void getMaxIndex(float *array, bool *visited, size_t start, size_t size, int *index){
	float max=FLT_MIN;
	for(int i=0; i<size; i++) {
		if(max<array[start+i] && visited[start+i]==false) {
			max=array[start+i];
			(*index)=i;
		}
	}
}

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
__global__
void widestPathKernel(int *offsets, int *destination_indices, int *edge_ids, float** variable_edges, int *nodes_types, int* path, int* path_edge, bool *visited, float *weights, int *discount, int *all_hosts_index, float* result, unsigned int hosts_size, unsigned int nodes_size){
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	if(x >= hosts_size-1 || y >= hosts_size || x>=y) return;

	int src = all_hosts_index[x];
	int dst = all_hosts_index[y];
	/*
	    Exemple: 6 hosts -> The array is composed by :
	    0-01 1-02 2-03 3-04 4-05
	    5-12 6-13 7-14 8-15
	    9-23 10-24 11-25
	    12-34 13-35
	    14-45

	   The matrix has [(|hosts_size|*(|hosts_size|-1))/2]*nodes_size
	 */

	int initial_index = nodes_size*(x* hosts_size + y   - discount[x]);
	printf("THREAD X:%d Y%d RUNNING WITH SRC: %d DST %d INITIAL INDEX %d\n",x,y,src,dst,initial_index);

	int node_index, next_node=0;
	size_t destination_index=0;
	float next_node_weight=0, alt=0;

	for(size_t i=0; i<nodes_size; i++) {
		weights[initial_index+i]=-1;
		path[initial_index+i]=-1;
		path_edge[initial_index+i]=-1;
		result[initial_index+i]=-1;
		visited[initial_index+i]=false;
	}

	weights[initial_index+src]=FLT_MAX;
	visited[initial_index+src]=false;
	visited[initial_index+dst]=false;

	while(true) {
		getMaxIndex(weights, visited, initial_index, nodes_size, &node_index);
		// printf("THREAD %d %d - NEXT NODE %d - %d - %d\n",x,y,next_node, initial_index+next_node,node_index);
		// printf("Node %d has Highest Weight %f\n",node_index, weights[node_index]);
		if(visited[initial_index+node_index]==true || node_index==dst) {
			// printf("Encerrei Cheguei ao Destino\n");
			break; //simulate empty queue or we found the destination node
		}
		visited[initial_index+node_index]=true;

		// actual_weight = weights[initial_index+next_node];
		// weights[initial_index+next_node]=FLT_MIN;


		// printf("Offset %d\n",offsets[node_index]);
		// need to run through all the edges of the Node_index
		for(destination_index= offsets[node_index];
		    destination_index< offsets[node_index+1];
		    destination_index++) {

			next_node = destination_indices[destination_index];
			if(visited[initial_index+next_node]) {
				// printf("PULEI ESSE NODE %d\n",initial_index+next_node);
				continue; //if the node is another host, ignore it.
			}
			// else printf("CONTINUANDO\n");

			next_node_weight = variable_edges[1][edge_ids[destination_index]]; // get the value of the edge bettween U and V.

			// printf("Agora vou olhar o NI %f E NN %f\n",weights[initial_index+node_index] < next_node_weight);

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
			// printf("FIQUEI COM %f\n",alt);
			// printf("ALT: %f WEIGHTS %f\n",alt,weights[initial_index+next_node]);
			if(alt> weights[initial_index+next_node]) {
				// printf("TROCANDO %f -> %f\n",weights[initial_index+next_node],alt);
				weights[initial_index+next_node] = alt;
				path[initial_index+next_node] = node_index;
				path_edge[initial_index+next_node] = edge_ids[destination_index];
			}else{
				// printf("Nao troquei o valor %f %f\n",alt, weights[initial_index+next_node]);
			}
		}
	}
	result[initial_index]=weights[initial_index+dst];
	printf("Kernel Result[%d]=%f\n\n",initial_index, result[initial_index]);
	printf("END KERNEL\n");
}
}

#endif
