#ifndef _SERIAL_DIJKSTRA_CUH
#define _SERIAL_DIJKSTRA_CUH

/*! \file
 *  \brief Serial Dijkstra Interface functions
 */

#include <vnegpu/graph.cuh>
#include <vnegpu/algorithm/serial/detail/dijkstra_imp.cuh>

namespace vnegpu
{
  namespace algorithm
  {
    namespace serial
    {
      /**
       * \brief Calculate the distance starting from a node on the graph.
       * \param initial_node The initial node to calculate the distance.
       * \param graph_distance The graph to be used.
       * \param distance_index The edge variable index to save the metric.
       * \param metric_functor The functor to be used to calculate.
       */
      template <typename T, class VariablesType, class MetricFunctor>
      void dijkstra(graph<T,VariablesType> *graph_distance, int initial_node, int distance_index, MetricFunctor metric_functor)
      {
        vnegpu::algorithm::serial::detail::dijkstra_imp(graph_distance, initial_node, distance_index, metric_functor);
      }
    }//end serial
  }//end algorithm
}//end vnegpu



#endif
