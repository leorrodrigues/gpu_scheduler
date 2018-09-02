#ifndef _SERIAL_K_MEANS_IMP_CUH
#define _SERIAL_K_MEANS_IMP_CUH

/*! \file
 *  \brief K-Means Core/Implementation functions
 */

#include <limits>

#include <vnegpu/graph.cuh>
#include <vnegpu/util/host_matrix.cuh>

namespace vnegpu
{
  namespace algorithm
  {
    namespace serial
    {
      namespace detail
      {

        template <typename T, class VariablesType, class DistanceFunctor>
        void update_node_centers(graph<T,VariablesType> *graph_group, int number_clusters, DistanceFunctor distance_functor, int* centers, vnegpu::util::host_matrix<T>* distance_matrix){
            //printf("OI:%d\n",graph_group->get_num_nodes());
            for(int i=0; i<graph_group->get_num_nodes(); i++){
                //printf("A\n");
                T menor_distancia = std::numeric_limits<T>::max();
                //printf("B\n");
                int id_menor=-1;
                for(int y=0; y < number_clusters; y++){
                  //printf("Y:%d\n",y);
                  T distancia = distance_functor(distance_matrix, i, centers[y]);
                  //printf("Y:%dH\n",y);
                  if(distancia < menor_distancia){
                    menor_distancia = distancia;
                    id_menor = y;
                  }
                  //printf("Y:%dG\n",y);
                }
                //printf("C\n");
                graph_group->set_group_id(i, id_menor);
                //printf("Set[%d]=%d\n",i,id_menor);
            }
        }

        template <typename T, class VariablesType, class DistanceFunctor>
        void update_centers(graph<T,VariablesType> *graph_group, int number_clusters, DistanceFunctor distance_functor, int* centers, bool* running, vnegpu::util::host_matrix<T>* distance_matrix){

            int* temp_centers = (int*)malloc(sizeof(int)*number_clusters);

            T* temp_centers_distances = (T*)malloc(sizeof(T)*number_clusters);

            for(int i=0; i<number_clusters; i++){
              temp_centers_distances[i]=std::numeric_limits<T>::max();
            }

            for(int i=0; i<graph_group->get_num_nodes(); i++){
                int distancia_sum = 0;
                int self_group = graph_group->get_group_id(i);
                for(int y=0; y<graph_group->get_num_nodes(); y++){
                    if(self_group == graph_group->get_group_id(y)){
                       distancia_sum+=distance_functor(distance_matrix, i, y);
                    }
                }

                if(distancia_sum < temp_centers_distances[self_group]){
                  temp_centers_distances[self_group] = distancia_sum;
                  temp_centers[self_group] = i;
                }

            }

            for(int i=0; i<number_clusters; i++){
              if(temp_centers[i]!=centers[i]){
                centers[i]=temp_centers[i];
                *running=true;
              }
            }

        }


        /**
         * \brief Classify the nodes of the graph in groups based on a distance.
         * \param graph_distance The graph to be used.
         * \param group_index Th.
         */
        template <typename T, class VariablesType, class DistanceFunctor>
        void k_means_imp(graph<T,VariablesType> *graph_group, int number_clusters, DistanceFunctor distance_functor, vnegpu::util::host_matrix<T>* distance_matrix)
        {

          graph_group->initialize_group();

          int* centers = (int*)malloc(sizeof(int)*number_clusters);

          if(number_clusters > graph_group->get_num_nodes()){
            return;
          }

          for(int i=0; i<number_clusters; i++){
            centers[i] = rand() % graph_group->get_num_nodes();
            for(int y=0; y<i; y++){
              if(centers[i]==centers[y]){
                i--;
                break;
              }
            }
          }

          /*for(int i=0; i<number_clusters; i++){
            printf("[%d]=%d\n",i,centers[i]);
          }*/

          bool running = true;
          //int x =0;
          while(running){
              running=false;
              //printf("it=%d\n",x++);
              update_node_centers(graph_group, number_clusters, distance_functor, centers, distance_matrix);
              update_centers(graph_group, number_clusters, distance_functor, centers, &running, distance_matrix);
          }

          /*for(int i=0; i<number_clusters; i++){
            printf("[%d]=%d\n",i,centers[i]);
          }*/

        }
      }
    }//end detail
  }//end algorithm
}//end vnegpu



#endif
