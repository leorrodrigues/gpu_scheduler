#ifndef _SERIAL_MCL_CUH
#define _SERIAL_MCL_CUH

/*! \file
 *  \brief Serial MCL Interface functions
 */

#include <vnegpu/graph.cuh>
#include <vnegpu/algorithm/serial/detail/mcl_imp.cuh>

namespace vnegpu
{
  namespace algorithm
  {
    namespace serial
    {
      /**
       * \brief Classify the nodes of the graph in groups based on a distance.
       * \param graph_distance The graph to be used.
       */
      template <typename T, class VariablesType>
      void mcl(graph<T,VariablesType> *graph_group, int distance_index, float p_factor=2, float r_factor=1.4, float max_error=0.001)
      {
        vnegpu::algorithm::serial::detail::mcl_imp(graph_group, distance_index, p_factor, r_factor, max_error);
      }
    }
  }//end algorithm
}//end vnegpu



#endif
