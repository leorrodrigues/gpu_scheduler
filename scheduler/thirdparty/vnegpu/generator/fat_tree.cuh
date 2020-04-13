#ifndef _FAT_TREE_CUH
#define _FAT_TREE_CUH

/*! \file
 *  \brief The generator of fat-trees.
 */

#include <stdio.h>

#include "../graph.cuh"

namespace vnegpu {

namespace generator {
/**
 * \brief Generate a fat tree topology.
 * \param k_factor The k param of the fat tree.
 * \return The new graph.
 */
template <typename T, class VariablesType=vnegpu::graph_type::minimalist>
vnegpu::graph<T,VariablesType>* fat_tree(int k_factor){

	if(k_factor % 2 == 1)
	{
		throw std::invalid_argument("Fat-Tree Invalid hosts number.");
	}

	int k = k_factor;
	int k2=k/2;
	int nos = (k*k*k+5*k*k)/4;
	int enlaces = 3*(k2)*(k2)*k;
	int kk = (k2)*(k2);
	int hosts = 0;

	vnegpu::graph<T,VariablesType>* fat_tree = new vnegpu::graph<T,VariablesType>(nos, enlaces, VariablesType::total_nodes_variables, VariablesType::total_edges_variables);

	//For K Pods
	for(int i=0; i<k; i++)
	{
		int pod_ini=i*((5*k*k)/4);
		int pod_d_ini=i*(kk+k);

		//For kk Hosts in the Pod
		for(int y=0; y<kk; y++)
		{
			fat_tree->set_source_offsets(y + pod_d_ini, y + pod_ini);
			fat_tree->set_node_type(y + pod_d_ini, vnegpu::TYPE_HOST);
			hosts++;

			//Each host have 1 link to lvl1 switch
			fat_tree->set_destination_indices(y + pod_ini, pod_d_ini + kk + (y / k2) );
		}

		//For k2 switchs on lvl1
		for(int y=0; y<(k2); y++)
		{
			fat_tree->set_source_offsets(y + kk + pod_d_ini, y * k + kk + pod_ini);
			fat_tree->set_node_type(y + kk + pod_d_ini, vnegpu::TYPE_SWITH);

			//link with hosts
			for(int z=0; z<(k2); z++)
			{

				fat_tree->set_destination_indices( y * k + kk + pod_ini + z,  y * k2 + pod_d_ini + z);
			}

			//link with lvl2
			for(int z=0; z<(k2); z++)
			{
				fat_tree->set_destination_indices( y * k + kk + pod_ini + k2 + z, kk + k2 + pod_d_ini + z);
			}
		}

		//For k2 Switchs on lvl2
		for(int y=0; y<(k2); y++)
		{
			fat_tree->set_source_offsets(y + k2 + kk + pod_d_ini, k * k2 + y * k + kk + pod_ini);
			fat_tree->set_node_type(y + k2 + kk + pod_d_ini, vnegpu::TYPE_SWITH);

			//link witch lvl1
			for(int z=0; z<(k2); z++)
			{
				fat_tree->set_destination_indices(k * k2 + y * k + kk + pod_ini + z, kk + pod_d_ini + z);
			}

			//link with global switchs
			for(int z=0; z<(k2); z++)
			{
				fat_tree->set_destination_indices(k * k2 + y * k + kk + pod_ini + z + k2, k * (kk + k) + z + y * k2);
			}
		}

	}

	//For kk global switchs
	for(int i=0; i<kk; i++)
	{
		fat_tree->set_source_offsets(k * (kk + k) + i, i * k + k * (5 * k * k / 4) );
		fat_tree->set_node_type(k * (kk + k) + i, vnegpu::TYPE_SWITH_CORE);

		//All links with global switchs
		for(int y=0; y<k; y++)
		{
			fat_tree->set_destination_indices(i * k + k* (5 * k * k / 4) + y, y * (kk + k) + kk + k2 + i / k2 );
		}
	}

	fat_tree->set_source_offsets(nos, enlaces*2);

	fat_tree->check_edges_ids();
	fat_tree->set_hosts(hosts);

	//printf("fhosts:%d\n", hosts);

	return fat_tree;
}

}  //end generator

}//end vnegpu

#endif
