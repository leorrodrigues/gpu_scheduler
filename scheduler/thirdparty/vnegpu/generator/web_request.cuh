#ifndef _WEB_REQUEST_GEN_CUH
#define _WEB_REQUEST_GEN_CUH

/*! \file
 *  \brief The generator of web requests.
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
    vnegpu::graph<T,VariablesType>* web_request(int work_layer, int db_layer){


      int nos = 3 + work_layer + db_layer;
      int enlaces = 1 + 2*work_layer + db_layer;

      vnegpu::graph<T,VariablesType>* request = new vnegpu::graph<T,VariablesType>(nos, enlaces, VariablesType::total_nodes_variables, VariablesType::total_edges_variables);

      request->set_source_offsets(0, 0);
      request->set_node_type(0, vnegpu::TYPE_HOST);
      request->set_destination_indices(0, 1);

      request->set_source_offsets(1, 1);
      request->set_node_type(1, vnegpu::TYPE_SWITH);
      request->set_destination_indices(1, 0);

      for(int i=0; i<work_layer; i++){
          request->set_destination_indices(2+i, 2+i);
      }

      for(int i=0; i<work_layer; i++){
          request->set_source_offsets(2+i, 2+work_layer+2*i);
          request->set_node_type(2+i, vnegpu::TYPE_HOST);
          request->set_destination_indices(2+work_layer+2*i, 1);
          request->set_destination_indices(2+work_layer+2*i+1, work_layer+2);
      }

      request->set_source_offsets(work_layer+2, 2+3*work_layer);
      request->set_node_type(work_layer+2, vnegpu::TYPE_SWITH);

      for(int i=0; i<work_layer; i++){
        request->set_destination_indices(2+3*work_layer+i, 2+i);
      }

      for(int i=0; i<db_layer; i++){
        request->set_destination_indices(2+4*work_layer+i, 3+work_layer+i);
      }

      for(int i=0; i<db_layer; i++){
        request->set_source_offsets(3+work_layer+i, 2+4*work_layer+db_layer+i);
        request->set_node_type(3+work_layer+i, vnegpu::TYPE_HOST);
        request->set_destination_indices(2+4*work_layer+db_layer+i, work_layer+2);
      }


      //Finish the CSR strute
      request->set_source_offsets(nos,enlaces*2);

      /*printf("Resource Offset\n");
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
      request->check_edges_ids();

      return request;
    }
  }//end generator
}//end vnegpu

#endif
