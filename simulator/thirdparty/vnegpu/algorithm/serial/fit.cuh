#ifndef _SERIAL_FIT_CUH
#define _SERIAL_FIT_CUH

/*! \file
 *  \brief Serial Fit Interface functions
 */

#include <vnegpu/graph.cuh>
#include <vnegpu/algorithm/serial/detail/fit_imp.cuh>

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
      template <typename T, class VariablesType, class AlocationFunctor>
      void fit(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, AlocationFunctor func, bool only_nodes=false)
      {
        vnegpu::algorithm::serial::detail::fit_imp(data_center, request, func);
      }
    }//end serial
  }//end algorithm
}//end vnegpu



#endif
