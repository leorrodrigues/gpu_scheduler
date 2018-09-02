#ifndef _DIJKSTRA_IMP_CUH
#define _DIJKSTRA_IMP_CUH

/*! \file
 *  \brief Dijkstra Core/Implementation functions
 */
#include <limits>

#include <vnegpu/graph.cuh>

#include <vnegpu/util/util.cuh>

namespace vnegpu
{
  namespace algorithm
  {
    namespace detail
    {
      /**
       * \brief Initialize the algorithm required variables.
       * \param graph_distance The graph to be used.
       * \param distance_index The node variable index to store the distance.
       * \param d_distance_temp Temporary array to be used to store distance.
       * \param infinite The max distance that can be stored.
       * \param initial_node The initial node to calculate the distance.
       * \param d_frontier The frontier of the algorithm.
       */
      template <typename T, class VariablesType>
      __global__ void
      initialize_dijkstra(vnegpu::graph<T,VariablesType> graph_distance,
                          int distance_index,
                          T* d_distance_temp,
                          bool* d_frontier,
                          T infinite,
                          int initial_node)
      {
        int id = threadIdx.x + blockIdx.x * blockDim.x;

        if(id >= graph_distance.get_num_nodes())
             return;

        if(initial_node==id)
        {
          graph_distance.set_variable_node(distance_index,id,0);
          d_distance_temp[id]=0;
          d_frontier[id]=1;
        }else{
          graph_distance.set_variable_node(distance_index,id,infinite);
          d_distance_temp[id]=infinite;
          d_frontier[id]=0;
        }

      }

      /**
       * \brief Calculate an iteration of the algorithm.
       * \param graph_distance The graph to be used.
       * \param distance_index The node variable index to store the distance.
       * \param d_distance_temp Temporary array to be used to store distance.
       * \param d_done Bool to check if the algorithm ends.
       * \param d_frontier The frontier of the algorithm.
       * \param metric_functor The functor to be used to calculate.
       */
      template <typename T, class VariablesType, class MetricFunctor>
      __global__ void
      dijkstra_iteration(vnegpu::graph<T,VariablesType> graph_distance,
                         T* d_distance_temp,
                         bool* d_frontier,
                         bool* d_done,
                         MetricFunctor metric_functor,
                         int distance_index)
      {
        int id = threadIdx.x + blockIdx.x * blockDim.x;

        if(id==0){
          *d_done = true;
        }

        if(id >= graph_distance.get_num_nodes())
             return;

        if(d_frontier[id]){
          d_frontier[id]=false;
          T self_distance = graph_distance.get_variable_node(distance_index,id);

          for(int i=graph_distance.get_source_offset(id); i<graph_distance.get_source_offset(id+1); i++)
          {
              T distance = metric_functor(&graph_distance,i,id);
              int neighbor = graph_distance.get_destination_indice(i);
              atomicMin(&d_distance_temp[neighbor], self_distance+distance);
          }

        }

      }

      /**
       * \brief Update the distance values on the graph struture and check if a new iteration is needed.
       * \param graph_distance The graph to be used.
       * \param distance_index The node variable index to store the distance.
       * \param d_distance_temp Temporary array to be used to store distance.
       * \param d_frontier The frontier of the algorithm.
       * \param d_done Bool to check if the algorithm ends.
       */
      template <typename T, class VariablesType>
      __global__ void
      dijkstra_save(vnegpu::graph<T,VariablesType> graph_distance,
                                    T* d_distance_temp,
                                    bool* d_frontier,
                                    bool* d_done,
                                    int distance_index)
      {
        int id = threadIdx.x + blockIdx.x * blockDim.x;

        if(id >= graph_distance.get_num_nodes())
             return;

        if(graph_distance.get_variable_node(distance_index,id) > d_distance_temp[id]){

          d_frontier[id]=true;
          *d_done = false;
          graph_distance.set_variable_node(distance_index, id, d_distance_temp[id]);
        }
        d_distance_temp[id] = graph_distance.get_variable_node(distance_index,id);

      }

      /**
       * \brief Calculate the distance starting from a node on the graph.
       * \param initial_node The initial node to calculate the distance.
       * \param graph_distance The graph to be used.
       * \param distance_index The edge variable index to save the metric.
       * \param metric_functor The functor to be used to calculate.
       */
      template <typename T, class VariablesType, class MetricFunctor>
      void dijkstra_imp(graph<T,VariablesType> *graph_distance, int initial_node, int distance_index, MetricFunctor metric_functor)
      {
        int num = graph_distance->get_num_nodes()/CUDA_BLOCK_SIZE + 1;

        dim3 Block(CUDA_BLOCK_SIZE);
        dim3 Grid(num);

        T infinite = std::numeric_limits<T>::max();

        T* d_distance_temp;
        cudaMalloc(&d_distance_temp, sizeof(T) * graph_distance->get_num_nodes() );

        bool* d_frontier;
        cudaMalloc(&d_frontier, sizeof(bool) * graph_distance->get_num_nodes() );

        initialize_dijkstra<<<Grid, Block>>>(*graph_distance, distance_index, d_distance_temp, d_frontier, infinite, initial_node);

        bool done = false;
        bool* d_done;

        cudaMalloc(&d_done, sizeof(bool));

        while(!done){
          dijkstra_iteration<<<Grid, Block>>>(*graph_distance, d_distance_temp, d_frontier, d_done, metric_functor, distance_index);
          dijkstra_save<<<Grid, Block>>>(*graph_distance, d_distance_temp, d_frontier, d_done, distance_index);
          cudaMemcpy(&done,d_done,sizeof(bool),cudaMemcpyDeviceToHost);
        }

        cudaFree(d_distance_temp);
        cudaFree(d_frontier);
        cudaFree(d_done);

      }
    }//end detail
  }//end algorithm
}//end vnegpu



#endif
