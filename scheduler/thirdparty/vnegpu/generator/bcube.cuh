#ifndef _BCUBE_CUH
#define _BCUBE_CUH

/*! \file
 *  \brief The generator of bcubes.
 */

#include <stdio.h>

#include "../graph.cuh"

#include "../util/util.cuh"

namespace vnegpu
{
namespace generator
{
/**
 * \brief Generate a bcube topology.
 * \param num_hosts_0_level The number of hosts per bcube0.
 * \param num_levels_switchs The number of levels.
 * \return The new graph.
 */
template <typename T, class VariablesType=vnegpu::graph_type::minimalist>
vnegpu::graph<T,VariablesType>* bcube(int num_hosts_0_level, int num_levels_switchs){

	int k = num_levels_switchs;
	int nu = num_hosts_0_level;
	int hosts = pow(nu,k+1);
	int swithPerLevel = pow(nu,k);
	int nos = hosts+swithPerLevel*(k+1);
	int enlaces = pow(nu,k+1)*(k+1);

	//Create the new graph.
	vnegpu::graph<T,VariablesType>* bcube = new vnegpu::graph<T,VariablesType>(nos, enlaces, VariablesType::total_nodes_variables, VariablesType::total_edges_variables);

	//Create the links between the hosts and switchs
	for(int i=0; i<hosts; i++)
	{
		bcube->set_source_offsets(i,i*(k+1));
		bcube->set_node_type(i, vnegpu::TYPE_HOST);

		for(int y=0; y<(k+1); y++)
		{
			bcube->set_destination_indices(i*(k+1)+y,i%ipow(nu,y)+i/ipow(nu,y+1)*ipow(nu,y)+hosts+swithPerLevel*y);
		}
	}

	//Create the links between the switchs and hosts
	for(int i=0; i<(k+1); i++)
	{
		for(int z=0; z<swithPerLevel; z++)
		{
			bcube->set_source_offsets(hosts+swithPerLevel*i+z,(hosts*(k+1))+(swithPerLevel*i+z)*nu);
			bcube->set_node_type(hosts+swithPerLevel*i+z, vnegpu::TYPE_SWITH);

			for(int y=0; y<nu; y++)
			{
				bcube->set_destination_indices((hosts*(k+1))+(swithPerLevel*i+z)*nu+y, z/ipow(nu,i)*ipow(nu,i+1)+(i==0 ? 0 : (z%ipow(nu,i)))+y*ipow(nu,i) );
			}
		}
	}

	//Finish the CSR strute
	bcube->set_source_offsets(nos,enlaces*2);

	//Create the information for the undirected graph
	bcube->check_edges_ids();
	bcube->set_hosts(hosts);

	return bcube;
}
}  //end generator
}//end vnegpu

#endif
