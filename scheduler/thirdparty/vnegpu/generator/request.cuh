#ifndef _REQUEST_GEN_CUH
#define _REQUEST_GEN_CUH

/*! \file
 *  \brief The generator of requests.
 */

#include <stdio.h>

#include <vnegpu/graph.cuh>

namespace vnegpu
{
  namespace generator
  {
    /**
     * \brief Generate a request topology.
     * \param num_hosts_0_level The number of hosts per bcube0.
     * \param num_levels_switchs The number of levels.
     * \return The new graph.
     */
    template <typename T, class VariablesType=vnegpu::graph_type::minimalist>
    vnegpu::graph<T,VariablesType>* request_gen(int num_levels, int* nodes_per_level){

      int nos_anterior = 1;
      int nos = 1;
      int enlaces = 0;
      for(int i=0; i<num_levels; i++){
        nos_anterior *= nodes_per_level[i];
        nos+=nos_anterior;
        enlaces+=nos_anterior;
      }

      //printf("nos:%d\n",nos);
      //printf("enlaces:%d\n",enlaces);

      //Create the new graph.
      vnegpu::graph<T,VariablesType>* request = new vnegpu::graph<T,VariablesType>(nos, enlaces, VariablesType::total_nodes_variables, VariablesType::total_edges_variables);

      int nos_in_this_level = 1;
      int start_anterior=0;
      int start=1;
      int enlace_atual=0;
      int offset_atual=0;
      int no_atual=1;

      request->set_source_offsets(0, offset_atual);
      request->set_node_type(0, vnegpu::TYPE_SWITH);
      offset_atual=nodes_per_level[0];
      for(int i=1; i<=nodes_per_level[0]; i++){
          request->set_destination_indices(enlace_atual++, i);
      }

      for(int i=0; i<num_levels; i++){

        nos_in_this_level *= nodes_per_level[i];
        start+=nos_in_this_level;

        for(int y=0; y<nos_in_this_level; y++){
            if( (i+1) < num_levels){
                request->set_node_type(no_atual, vnegpu::TYPE_SWITH);
                request->set_source_offsets(no_atual++, offset_atual);
                offset_atual +=1+nodes_per_level[i+1];
            }else{
                request->set_node_type(no_atual, vnegpu::TYPE_HOST);
                request->set_source_offsets(no_atual++, offset_atual);
                offset_atual +=1;
            }

            request->set_destination_indices(enlace_atual++, start_anterior+y/nodes_per_level[i]);
            if( (i+1) < num_levels){
                for(int z=0; z<nodes_per_level[i+1]; z++){
                  //printf("%d -> %d com start:%d, y:%d, no:%d, z:%d\n", enlace_atual, start+y*nodes_per_level[i+1]+z, start, y, nodes_per_level[i+1], z);
                  request->set_destination_indices(enlace_atual++, start+y*nodes_per_level[i+1]+z);
                }
            }
        }

        if(i==0){
          start_anterior=1;
        }else{
          start_anterior+=nodes_per_level[i-1];
        }

      }



      //Finish the CSR strute
      request->set_source_offsets(nos,enlaces*2);
      /*
      printf("Resource Offset\n");
      for(int i=0; i<=nos; i++){
        printf("[%d]=%d", i, request->get_source_offset(i));
      }
      printf("\n");

      printf("Destination Indice\n");
      for(int i=0; i<enlaces*2; i++){
        printf("[%d]=%d", i, request->get_destination_indice(i));
      }
      printf("\n");
      */
      //Create the information for the undirected graph
      request->check_edges_ids();

      return request;
    }
  }//end generator
}//end vnegpu

#endif
