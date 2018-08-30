#ifndef _LOCAL_METRICS_IMP_CUH
#define _LOCAL_METRICS_IMP_CUH

/*! \file
 *  \brief Local Metrics Core/Implementation functions
 */

#include <vnegpu/graph.cuh>

namespace vnegpu
{
  namespace algorithm
  {
    namespace detail
    {
      /**
       * \brief Kernel for local metric.
       * \param graph_metric The graph to be used.
       * \param rank_index The node index to save the metric.
       * \param metric_functor The functor to be used to calculate.
       */
      template <typename T, class VariablesType, class MetricFunctor>
      __global__ void calculate_local_metric(vnegpu::graph<T,VariablesType> graph_metric, int rank_index, MetricFunctor metric_functor)
      {
        int id = threadIdx.x + blockIdx.x * blockDim.x;

        if(id >= graph_metric.get_num_nodes())
             return;

        //Calculate and update the variable
        graph_metric.set_variable_node(rank_index, id, metric_functor(&graph_metric,0,id) );
      }

      /**
       * \brief Apply the local metric on the graph.
       * \param graph_metric The graph to be used.
       * \param rank_index The node index to save the metric.
       * \param metric_functor The functor to be used to calculate.
       */
      template <typename T, class VariablesType, class MetricFunctor>
      void local_metric_imp(graph<T,VariablesType> *graph_metric, int rank_index, MetricFunctor metric_functor)
      {
        if(rank_index >= graph_metric->get_num_var_nodes()){
          throw std::invalid_argument("The Rank Index is invalid.");
        }

        int num = graph_metric->get_num_nodes()/CUDA_BLOCK_SIZE + 1;
        dim3 Block(CUDA_BLOCK_SIZE);
        dim3 Grid(num);

        //Call the kernel
        calculate_local_metric<<<Grid, Block>>>(*graph_metric, rank_index, metric_functor); CUDA_CHECK();

        //Sync before the function end.
        cudaDeviceSynchronize();
      }
    }//end detail
  }//end algorithm
}//end vnegpu

#endif
