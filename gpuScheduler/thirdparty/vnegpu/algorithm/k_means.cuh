#ifndef _K_MEANS_CUH
#define _K_MEANS_CUH

/*! \file
 *  \brief K-Means Interface functions
 */

#include <vnegpu/graph.cuh>
#include <vnegpu/algorithm/detail/k_means_imp.cuh>

namespace vnegpu
{
  namespace algorithm
  {
    /**
     * \brief Classify the nodes of the graph in groups based on a distance.
     * \param graph_distance The graph to be used.
     * \param group_index The index of the nodes to store groups.
     */
    template <typename T, class VariablesType, class DistanceFunctor>
    void k_means(graph<T,VariablesType> *graph_group, int number_clusters, DistanceFunctor metric_functor)
    {
      vnegpu::algorithm::detail::k_means_imp(graph_group, number_clusters, metric_functor);
    }

  }//end algorithm
}//end vnegpu



#endif
