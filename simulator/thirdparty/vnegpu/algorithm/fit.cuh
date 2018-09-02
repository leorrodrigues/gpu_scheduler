#ifndef _FIT_CUH
#define _FIT_CUH

/*! \file
 *  \brief Fit Interface functions
 */

#include <vnegpu/graph.cuh>
#include <vnegpu/algorithm/detail/fit_imp.cuh>

namespace vnegpu
{
  namespace algorithm
  {
    /**
     * \brief Calculate the distance starting from a node on the graph.
     * \param initial_node The initial node to calculate the distance.
     * \param graph_distance The graph to be used.
     * \param distance_index The edge variable index to save the metric.
     * \param metric_functor The functor to be used to calculate.
     */
    template <typename T, class VariablesType, class AlocationFunctor>
    vnegpu::fit_return fit(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, AlocationFunctor func, bool only_nodes=false)
    {
      return vnegpu::algorithm::detail::fit_imp(data_center, request, func, only_nodes);
    }

  }//end algorithm
}//end vnegpu



#endif
