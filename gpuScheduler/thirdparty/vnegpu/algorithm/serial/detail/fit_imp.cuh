#ifndef _SERIAL_FIT_IMP_CUH
#define _SERIAL_FIT_IMP_CUH

/*! \file
 *  \brief Serial Fit Core/Implementation functions
 */

#include <limits>
#include <queue>
#include <vector>
#include <functional>

#include <vnegpu/graph.cuh>
#include <vnegpu/util/util.cuh>

namespace vnegpu
{
  namespace algorithm
  {
    namespace serial
    {
      namespace detail
      {

        template <typename T, class VariablesType, class AlocationFunctor>
        void dijkstra_imp_with_allocation_check(graph<T,VariablesType> *graph_distance,
                                                graph<T,VariablesType> *request,
                                                int initial_node,
                                                int edge_request,
                                                T* distance_index,
                                                AlocationFunctor alloc,
                                                int* path)
        {

          T infinite = std::numeric_limits<T>::max();

          std::pair<T, int> initial_pair(0.0f, request->get_allocation_to_nodes_ids(initial_node));

          for(int y=0; y<graph_distance->get_num_nodes(); y++){
              distance_index[y]=infinite;
          }
          distance_index[request->get_allocation_to_nodes_ids(initial_node)]=0;

          std::priority_queue<std::pair<T, int>, std::vector<std::pair<T, int>>, std::greater<std::pair<T, int>> > queue;
          queue.push(initial_pair);

          while(!queue.empty())
          {
            std::pair<T, int> element = queue.top();
            queue.pop();

            int self_distance = element.first;
            int node = element.second;

            if(self_distance > distance_index[node])
                continue;

            for(int i=graph_distance->get_source_offset(node); i<graph_distance->get_source_offset(node+1); i++)
            {
              if(alloc.is_edge_allocable(graph_distance, request, i, edge_request) )
              {
                T distance = alloc.edge_distance(graph_distance, i, node);
                int neighbor = graph_distance->get_destination_indice(i);
                T new_distance = self_distance+distance;
                if(new_distance < distance_index[neighbor]){
                    distance_index[neighbor] = new_distance;
                    for(int x=graph_distance->get_source_offset(neighbor); x<graph_distance->get_source_offset(neighbor+1); x++)
                    {
                      int neighbor_x = graph_distance->get_destination_indice(x);
                      if(neighbor_x == node){
                        path[neighbor]=x;
                      }
                    }
                    std::pair<T, int> new_pair(new_distance, neighbor);
                    queue.push(new_pair);
                }
              }
            }
          }

          for(int y=0; y<graph_distance->get_num_nodes(); y++){
              //printf("path[%d]=%d\n",y,path[y]);
              //printf("distance_index[%d]=%f\n",y,distance_index[y]);
          }

        }

        template <typename T, class VariablesType, class AlocationFunctor>
        void alloc_node(graph<T,VariablesType>* data_center,
                                  graph<T,VariablesType>* request,
                                  int request_id,
                                  AlocationFunctor func)
        {

          int my_best=-1;
          for(int i=0;i<data_center->get_num_nodes();i++){
            if(func.is_node_allocable(data_center, request, i, request_id))
            {
              if(my_best==-1 || func.is_best_node(data_center, i, my_best) ){
                my_best=i;
              }
            }
          }

          func.alloc_node(data_center, request, my_best, request_id);
          request->set_allocation_to_nodes_ids(request_id, my_best);
          //printf("Node Request: %d allocated on Data Center Node: %d\n",request_id,my_best);

        }

        template <typename T, class VariablesType, class AlocationFunctor>
        void alloc_path(graph<T,VariablesType>* data_center,
                       graph<T,VariablesType>* request,
                       int edge_request,
                       int* path,
                       int origim,
                       int destiny,
                       AlocationFunctor func)
        {
          int node = request->get_allocation_to_nodes_ids(destiny);
          int real_origin = request->get_allocation_to_nodes_ids(origim);
          //verify if INF
          int edge = path[node];
          if(node==real_origin){
            //printf("SAME NODE CONDITION\n");
            return;
          }
          if(edge==-1){
            //printf("ERROR ALOC NO PATH %d %d->%d\n",node,origim,destiny);
            return;
          }
          int next_node = data_center->get_destination_indice(edge);
          int it=100;
          while(next_node != real_origin && it--){
            func.alloc_edge(data_center, request, edge, edge_request);
            //request->set_allocation_to_edges_ids(edge_request, edge);
            node=next_node;
            edge = path[node];
            if(edge==-1){
              //printf("ERROR ALOC NO PATH2\n");
              return;
            }
            //printf("Next:%d-%d\n",next_node, edge);
            next_node = data_center->get_destination_indice(edge);
          }
          if(it<1){
            //printf("ERRAO\n");
          }
          func.alloc_edge(data_center, request, edge, edge_request);
          //request->set_allocation_to_edges_ids(edge_request, edge);
        }

        /**
         * \brief Classify the nodes of the graph in groups based on a distance.
         * \param graph_distance The graph to be used.
         * \param group_index Th.
         */
         template <typename T, class VariablesType, class AlocationFunctor>
         void fit_imp(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, AlocationFunctor func)
         {

           request->initialize_allocation();

           int* d_path  = (int*)malloc(sizeof(int) * data_center->get_num_edges() );

           T* d_temp_distance = (T*)malloc(sizeof(T) * data_center->get_num_edges() );



            for(int i=0;i<request->get_num_nodes();i++)
            {
              alloc_node(data_center, request, i, func);
            }

            bool* concluded_edges = (bool*)malloc(request->get_num_edges()*sizeof(bool));
            memset(concluded_edges,0,request->get_num_edges()*sizeof(bool));

            for(int id=0;id<request->get_num_nodes();id++)
            {
              for(int i=request->get_source_offset(id); i<request->get_source_offset(id+1); i++)
              {
                if(!concluded_edges[request->get_egdes_ids(i)])
                {
                  concluded_edges[request->get_egdes_ids(i)]=true;
                  int neighbor = request->get_destination_indice(i);

                  for(int z=0;z<data_center->get_num_nodes();z++)
                  {
                    d_path[z]=-1;
                  }
                  //printf("Processing Request edge %d->%d\n",id,neighbor);
                  vnegpu::algorithm::serial::detail::dijkstra_imp_with_allocation_check(data_center,
                                                                                         request,
                                                                                        id,
                                                                                        i,
                                                                                        d_temp_distance,
                                                                                        func,
                                                                                        d_path);
                  alloc_path(data_center, request, i, d_path, id, neighbor, func);
                }
              }
            }
         }

     }//end detail
    }//end serial
  }//end algorithm
}//end vnegpu



#endif
