#ifndef _MCL_CUH
#define _MCL_CUH

/*! \file
 *  \brief MCL Interface functions
 */

#include "../graph.cuh"
#include "detail/mcl_imp.cuh"

namespace vnegpu
{
namespace algorithm
{
/**
 * \brief Classify the nodes of the graph in groups based on a distance.
 * \param graph_distance The graph to be used.
 */
template <typename T, class VariablesType>
void mcl(graph<T,VariablesType> *graph_group, int distance_index, float p_factor=2, float r_factor=1.4, float max_error=0.001)
{
	vnegpu::algorithm::detail::mcl_imp(graph_group, distance_index, p_factor, r_factor, max_error);
}

}  //end algorithm
}//end vnegpu



#endif
