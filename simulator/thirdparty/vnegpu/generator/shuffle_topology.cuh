#ifndef _SHUFFLE_TOPOLOGY_CUH
#define _SHUFFLE_TOPOLOGY_CUH

/*! \file
 *  \brief The shuffle of topologies.
 */

#include <stdio.h>
#include <random>

#include <vnegpu/graph.cuh>

namespace vnegpu
{
  namespace generator
  {
    template <typename T, class VariablesType>
    void shuffle(vnegpu::graph<T,VariablesType>* graph, int seed=1){
        std::mt19937 gen(seed);
        std::uniform_int_distribution<> id_generator(0, graph->get_num_nodes()-1);

/*
        printf("Resource Offset\n");
        for(int i=0; i<=graph->get_num_nodes(); i++){
          printf("[%d]=%d", i, graph->get_source_offset(i));
        }
        printf("\n");

        printf("Destination Indice\n");
        for(int i=0; i<graph->get_num_edges()*2; i++){
          printf("[%d]=%d", i, graph->get_destination_indice(i));
        }
        printf("\n");
*/
        for(int pid=0; pid<graph->get_num_nodes(); pid++){



          int new_id  = id_generator(gen);
          int id = pid;
          if(new_id<id){
            int t = new_id;
            new_id=id;
            id=t;
          }

          int old_destination_start = graph->get_source_offset(id);
          int new_destination_start = graph->get_source_offset(new_id);

          int self_edges_number = graph->get_source_offset(id+1)-old_destination_start;
          int new_id_edges_number = graph->get_source_offset(new_id+1)-new_destination_start;

          //printf("Changing %d->%d, size1:%d size2:%d\n",id,new_id,self_edges_number,new_id_edges_number);

          if(self_edges_number==new_id_edges_number){
              for(int edge=0; edge<self_edges_number; edge++){
                  int destination_old = graph->get_destination_indice(old_destination_start+edge);
                  int destination_new = graph->get_destination_indice(new_destination_start+edge);
                  graph->set_destination_indices(old_destination_start+edge, destination_new);
                  graph->set_destination_indices(new_destination_start+edge, destination_old);

                  //int edge_id_old = graph->get_egdes_ids(old_destination_start+edge);
                  //int edge_id_new = graph->get_egdes_ids(new_destination_start+edge);
                  //graph->set_egdes_ids(old_destination_start+edge, edge_id_new);
                  //graph->set_egdes_ids(new_destination_start+edge, edge_id_old);
              }
          }else if(self_edges_number>new_id_edges_number){
            int diff = self_edges_number-new_id_edges_number;

            int* temp_old = new int[self_edges_number];
            //int* temp_old_edge = new int[self_edges_number];
            for(int edge=0; edge<self_edges_number; edge++){
                temp_old[edge] = graph->get_destination_indice(old_destination_start+edge);
                //temp_old_edge[edge] = graph->get_egdes_ids(old_destination_start+edge);
            }
            for(int edge=0; edge<new_id_edges_number; edge++){
                int destination_new = graph->get_destination_indice(new_destination_start+edge);
                graph->set_destination_indices(old_destination_start+edge, destination_new);

                //int edge_id_old = graph->get_egdes_ids(old_destination_start+edge);
                //int edge_id_new = graph->get_egdes_ids(new_destination_start+edge);
                //graph->set_egdes_ids(old_destination_start+edge, edge_id_new);
                //graph->set_egdes_ids(new_destination_start+edge, edge_id_old);
            }
            for(int edge=old_destination_start+new_id_edges_number; edge<new_destination_start; edge++){
              int destination_new = graph->get_destination_indice(edge+diff);
              //int edge_id_new = graph->get_egdes_ids(edge+diff);
              graph->set_destination_indices(edge, destination_new);
              //graph->set_egdes_ids(edge, edge_id_new);
            }

            for(int edge=0; edge<self_edges_number; edge++){
                graph->set_destination_indices(new_destination_start-diff+edge, temp_old[edge]);
                //graph->set_egdes_ids(edge, temp_old_edge[edge]);
            }

            for(int z=id+1;z<=new_id; z++){
              int old_offset=graph->get_source_offset(z);
              graph->set_source_offsets(z, old_offset-diff);
            }
          //********
          //other
          //*******
          }else if(self_edges_number<new_id_edges_number){
            int diff = new_id_edges_number-self_edges_number;

            int* temp_old = new int[new_id_edges_number];
            //int* temp_old_edge = new int[self_edges_number];

            for(int edge=0; edge<new_id_edges_number; edge++){
                temp_old[edge] = graph->get_destination_indice(new_destination_start+edge);
                //temp_old_edge[edge] = graph->get_egdes_ids(new_destination_start+edge);
            }
            for(int edge=0; edge<self_edges_number; edge++){
                int destination_new = graph->get_destination_indice(old_destination_start+edge);
                graph->set_destination_indices(graph->get_source_offset(new_id+1)-self_edges_number+edge, destination_new);

                //int edge_id_old = graph->get_egdes_ids(old_destination_start+edge);
                //int edge_id_new = graph->get_egdes_ids(new_destination_start+edge);
                //graph->set_egdes_ids(old_destination_start+edge, edge_id_new);
                //graph->set_egdes_ids(new_destination_start+edge, edge_id_old);
            }
            for(int edge=graph->get_source_offset(new_id)-1; edge>old_destination_start; edge--){
              int destination_new = graph->get_destination_indice(edge);
              //int edge_id_new = graph->get_egdes_ids(edge);
              graph->set_destination_indices(edge+diff, destination_new);
              //graph->set_egdes_ids(edge+diff, edge_id_new);
            }

            for(int edge=0; edge<new_id_edges_number; edge++){
                graph->set_destination_indices(old_destination_start+edge, temp_old[edge]);
                //graph->set_egdes_ids(edge, temp_old_edge[edge]);
            }

            for(int z=id+1;z<=new_id; z++){
              int old_offset=graph->get_source_offset(z);
              graph->set_source_offsets(z, old_offset+diff);
            }

          }
          for(int y=0; y<graph->get_num_edges()*2; y++){
            if(graph->get_destination_indice(y)==id)
            {
              graph->set_destination_indices(y, new_id);
            }
            else if(graph->get_destination_indice(y)==new_id)
            {
              graph->set_destination_indices(y, id);
            }
          }

          for(int var=0; var<graph->get_num_var_nodes(); var++){
              T old = graph->get_variable_node(var, id);
              T new_var = graph->get_variable_node(var, new_id);
              graph->set_variable_node(var, id, new_var);
              graph->set_variable_node(var, new_id, old);
          }

          int group = graph->get_node_type(id);
          graph->set_node_type(id, graph->get_node_type(new_id));
          graph->set_node_type(new_id, group);

        }
/*
        printf("Resource Offset\n");
        for(int i=0; i<=graph->get_num_nodes(); i++){
          printf("[%d]=%d", i, graph->get_source_offset(i));
        }
        printf("\n");

        printf("Destination Indice\n");
        for(int i=0; i<graph->get_num_edges()*2; i++){
          printf("[%d]=%d", i, graph->get_destination_indice(i));
        }
        printf("\n");
*/
        graph->check_edges_ids();

        //graph->save_to_gexf("a_shuffle.gexf");



    }
  }
}

#endif
