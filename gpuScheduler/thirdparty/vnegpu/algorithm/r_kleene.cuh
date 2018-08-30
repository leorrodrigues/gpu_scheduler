#ifndef _R_KLEENE_CUH
#define _R_KLEENE_CUH

/*! \file
 *  \brief R-Kleene Interface functions
 */

#include <vnegpu/graph.cuh>
#include <vnegpu/algorithm/detail/r_kleene_imp.cuh>

namespace vnegpu
{
  namespace algorithm
  {
    /**
     * \brief Calculate the distance from all nodes to all nodes.
     * \param graph_distance The graph to be used.
     * \param distance_index The index of the adge variable used as distance.
     */
    template <typename T, class VariablesType>
    void r_kleene(graph<T,VariablesType> *graph_distance, int distance_index)
    {
      vnegpu::algorithm::detail::r_kleene_imp(graph_distance, distance_index);
    }

  }//end algorithm
}//end vnegpu



#endif
