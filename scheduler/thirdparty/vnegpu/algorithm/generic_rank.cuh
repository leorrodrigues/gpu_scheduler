#ifndef _GENERIC_RANK_CUH
#define _GENERIC_RANK_CUH

/*! \file
 *  \brief Generic Rank Interface functions
 */

#include <vnegpu/graph.cuh>
#include <vnegpu/algorithm/detail/generic_rank_imp.cuh>

namespace vnegpu
{
  namespace algorithm
  {
    /**
     * \brief Apply the Generic Rank on the graph.
     * \param graph_rank The graph to be used.
     * \param rank_index The node index to save the rank.
     * \param metric_functor The functor to be used to calculate.
     * \param error_factor The error used to stop iterations.
     * \param p_factor The PageRank P factor.
     */
    template <typename T, class VariablesType, class MetricFunctor>
    void generic_rank(graph<T,VariablesType> *graph_rank, int rank_index, MetricFunctor metric_functor, T error_factor=RANK_MAX_ERROR, float p_factor=0.85)
    {
      vnegpu::algorithm::detail::generic_rank_imp(graph_rank, rank_index, metric_functor, p_factor, error_factor);
    }

  }//end algorithm
}//end vnegpu



#endif
