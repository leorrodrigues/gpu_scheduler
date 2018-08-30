#ifndef _K_MEANS_IMP_CUH
#define _K_MEANS_IMP_CUH

/*! \file
 *  \brief K-Means Core/Implementation functions
 */

#include <limits>

#include <vnegpu/graph.cuh>


namespace vnegpu
{
  namespace algorithm
  {
    namespace detail
    {

      template <typename T, class VariablesType, class DistanceFunctor>
      __global__
      void update_node_centers(graph<T,VariablesType> graph_group, int number_clusters, DistanceFunctor distance_functor, int* centers, bool* running, T infinety, vnegpu::util::matrix<T> distance_matrix){
          int id = threadIdx.x + blockIdx.x * blockDim.x;

          if(id >= graph_group.get_num_nodes())
               return;

          if(id==0){
            *running=false;
          }

          T menor_distancia = infinety;

          int id_menor=-1;
          for(int y=0; y < number_clusters; y++){
            int x =  centers[y];
            T distancia = distance_functor(&distance_matrix, id, x);
            if(distancia < menor_distancia){
              menor_distancia = distancia;
              id_menor = y;
            }
          }
          graph_group.set_group_id(id, id_menor);

      }

      template <typename T, class VariablesType, class DistanceFunctor>
      __global__
      void calculate_cluster_distances(graph<T,VariablesType> graph_group, DistanceFunctor distance_functor, T* d_sum_distancias, vnegpu::util::matrix<T> distance_matrix){

        int id = threadIdx.x + blockIdx.x * blockDim.x;

        if(id >= graph_group.get_num_nodes())
             return;

        int distancia_sum = 0;
        int self_group = graph_group.get_group_id(id);
        for(int y=0; y<graph_group.get_num_nodes(); y++){
            if(self_group == graph_group.get_group_id(y)){
               distancia_sum+=distance_functor(&distance_matrix, id, y);
            }
        }
        d_sum_distancias[id]=distancia_sum;

      }

      template <typename T, class VariablesType>
      __global__
      void update_centers(graph<T,VariablesType> graph_group, T* d_sum_distancias, int* centers, bool* running, T infinety){

        int id = blockIdx.x;

        T menor_distancia = d_sum_distancias[centers[id]];

        for(int y=0; y<graph_group.get_num_nodes(); y++){
          if(id == graph_group.get_group_id(y)){
            if(menor_distancia > d_sum_distancias[y]){
              menor_distancia = d_sum_distancias[y];
              if(centers[id] != y){
                centers[id] = y;
                *running=true;
              }
            }
          }
        }

      }


      /**
       * \brief Classify the nodes of the graph in groups based on a distance.
       * \param graph_distance The graph to be used.
       * \param group_index Th.
       */
      template <typename T, class VariablesType, class DistanceFunctor>
      void k_means_imp(graph<T,VariablesType> *graph_group, int number_clusters, DistanceFunctor distance_functor)
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

        int* d_centers;
        cudaMalloc(&d_centers, sizeof(int)*number_clusters);

        cudaMemcpy(d_centers, centers, sizeof(int)*number_clusters ,cudaMemcpyHostToDevice);

        T* d_sum_distancias;
        cudaMalloc(&d_sum_distancias, sizeof(T)*graph_group->get_num_nodes());

        bool* d_running;
        cudaMalloc(&d_running, sizeof(bool));

        int num = graph_group->get_num_nodes()/CUDA_BLOCK_SIZE + 1;

        dim3 Block(CUDA_BLOCK_SIZE);
        dim3 Grid(num);

        T inf = std::numeric_limits<T>::max();

        //for(int i=0; i<number_clusters; i++){
        //  printf("[%d]=%d\n",i,centers[i]);
        //}

        vnegpu::util::matrix<T>* distance_matrix = graph_group->get_distance_matrix();

        bool running = true;
        //int x =0;
        while(running){
            running=false;
            //printf("it=%d\n",x++);
            update_node_centers<<<Grid, Block>>>(*graph_group, number_clusters, distance_functor, d_centers, d_running, inf, *distance_matrix);
            calculate_cluster_distances<<<Grid, Block>>>(*graph_group, distance_functor, d_sum_distancias, *distance_matrix);
            update_centers<<<number_clusters, 1>>>(*graph_group, d_sum_distancias, d_centers, d_running, inf);
            cudaMemcpy(&running,d_running,sizeof(bool),cudaMemcpyDeviceToHost);
        }

        cudaDeviceSynchronize(); CUDA_CHECK();
        cudaFree(d_centers);
        cudaFree(d_sum_distancias);
        cudaFree(d_running);
        free(centers);

      }
    }//end detail
  }//end algorithm
}//end vnegpu



#endif
