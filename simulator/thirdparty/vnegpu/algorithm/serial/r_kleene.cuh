#ifndef _SERIAL_R_KLEENE_CUH
#define _SERIAL_R_KLEENE_CUH

/*! \file
 *  \brief Serial R-Kleene Interface functions
 */

#include <vnegpu/graph.cuh>
#include <vnegpu/util/host_matrix.cuh>
#include <vnegpu/algorithm/serial/detail/r_kleene_imp.cuh>

namespace vnegpu
{
  namespace algorithm
  {
    namespace serial
    {
      /**
       * \brief Calculate the distance from all nodes to all nodes.
       * \param graph_distance The graph to be used.
       * \param distance_index The index of the adge variable used as distance.
       */
      template <typename T, class VariablesType>
      vnegpu::util::host_matrix<T>* r_kleene(graph<T,VariablesType> *graph_distance, int distance_index)
      {
        return vnegpu::algorithm::serial::detail::r_kleene_imp(graph_distance, distance_index);
      }
    }//end serial
  }//end algorithm
}//end vnegpu



#endif
