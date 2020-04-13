#ifndef _DCELL_CUH
#define _DCELL_CUH

/*! \file
 *  \brief The generator of decells.
 */

#include <stdio.h>

#include "../graph.cuh"

#include "../util/util.cuh"

namespace vnegpu
{
namespace generator
{
template <typename T, class VariablesType=vnegpu::graph_type::minimalist>
void buildDCell(vnegpu::graph<T,VariablesType>* dcell, int* pref, int n, int l, int tl, int* Tv, int* G){
	int links_per_host = 1+tl;
	int hosts = Tv[tl];
	if(l==0) {
		for(int i=0; i<n; i++) {
			int id = 0;
			for(int le=0; le<tl; le++) {
				id += pref[le] * Tv[tl-1-le];
			}
			id+=i;
			int id_switch = Tv[tl]+(id / n);
			int id_switch_only = id / n;

			//printf("Host:%d Switch:%d\n", id, id_switch);
			dcell->set_destination_indices(id*links_per_host, id_switch);
			dcell->set_destination_indices(hosts*links_per_host+id_switch_only*n+i, id);
			//printf("D:%d, id:%d\n", hosts*links_per_host+id_switch_only*n+i, id);
		}
		return;
	}

	int* new_pref = (int*)malloc(tl*sizeof(int));
	for(int z=0; z<(tl-l); z++) {
		new_pref[z]=pref[z];
	}
	for(int i=0; i<G[l]; i++) {
		new_pref[tl-l] = i;
		buildDCell(dcell, new_pref, n, l-1, tl, Tv, G);
	}
	free(new_pref);

	for(int i=0; i<Tv[l-1]; i++) {
		for(int j=i+1; j<G[l]; j++) {
			int uid_1 = j-1;
			int uid_2 = i;
			int id_1 = 0;
			for(int le=0; le<(tl-l); le++) {
				id_1 += pref[le] * Tv[tl-1-le];
			}

			int id_2 = id_1;
			id_1 += Tv[l-1] * i;
			id_1 += uid_1;
			id_2 += Tv[l-1] * j;
			id_2 += uid_2;
			dcell->set_destination_indices(id_1*links_per_host+l, id_2);
			dcell->set_destination_indices(id_2*links_per_host+l, id_1);
			//printf("ID1:%d ID2:%d\n", id_1, id_2);
		}
	}
	return;
}
/**
 * \brief Generate a bcube topology.
 * \param num_hosts_0_level The number of hosts per bcube0.
 * \param num_levels The number of levels.
 * \return The new graph.
 */
template <typename T, class VariablesType=vnegpu::graph_type::minimalist>
vnegpu::graph<T,VariablesType>* dcell(int num_hosts_0_level, int num_levels){

	int* t = (int*)malloc( (num_levels+1) * sizeof(int));
	int* g = (int*)malloc( (num_levels+1) * sizeof(int));

	g[0] = 1;
	t[0] = num_hosts_0_level;

	for(int i=1; i<(num_levels+1); i++) {
		g[i] = t[i-1]+1;
		t[i] = g[i] * t[i-1];
	}

	int hosts = t[num_levels];
	int swiths = t[num_levels]/num_hosts_0_level;
	int nos = hosts+swiths;
	int enlaces_dir = hosts*2 + (num_levels*hosts);
	int enlaces = enlaces_dir/2;

	//Create the new graph.
	vnegpu::graph<T,VariablesType>* dcell = new vnegpu::graph<T,VariablesType>(nos, enlaces, VariablesType::total_nodes_variables, VariablesType::total_edges_variables);

	//printf("nos:%d\n",nos);
	//printf("enlaces:%d\n",enlaces);
	//printf("hosts:%d\n", hosts);

	int links_per_host = 1+num_levels;

	for(int i=0; i<hosts; i++)
	{
		dcell->set_source_offsets(i, i*links_per_host);
		dcell->set_node_type(i, vnegpu::TYPE_HOST);
	}

	int hosts_links = hosts*links_per_host;

	for(int i=0; i<swiths; i++)
	{
		dcell->set_source_offsets(i+hosts, hosts_links + i*num_hosts_0_level);
		dcell->set_node_type(i+hosts, vnegpu::TYPE_SWITH);
	}

	int* pref = (int*)malloc(num_levels*sizeof(int));
	buildDCell(dcell, pref, num_hosts_0_level, num_levels, num_levels, t, g);
	free(pref);

	//Finish the CSR strute
	dcell->set_source_offsets(nos,enlaces*2);
	/*
	   printf("Resource Offset\n");
	   for(int i=0; i<=nos; i++){
	   printf("[%d]=%d", i, dcell->get_source_offset(i));
	   }
	   printf("\n");

	   printf("Destination Indice\n");
	   for(int i=0; i<enlaces*2; i++){
	   printf("[%d]=%d", i, dcell->get_destination_indice(i));
	   }
	   printf("\n");
	 */
	//Create the information for the undirected graph
	dcell->check_edges_ids();
	dcell->set_hosts(hosts);

	return dcell;
}
}  //end generator
}//end vnegpu

#endif
