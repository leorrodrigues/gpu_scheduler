#ifndef _GRAPH_METRICS_CUH
#define _GRAPH_METRICS_CUH

/*! \file
 *  \brief Local Metrics Interface functions
 */

#include <vnegpu/graph.cuh>
#include <vnegpu/algorithm/detail/graph_metric_imp.cuh>

namespace vnegpu
{
  namespace algorithm
  {
    /**
     * \brief Apply the local metric on the graph.
     * \param graph_rank The graph to be used.
     * \param rank_index The node index to save the metric.
     * \param metric_functor The functor to be used to calculate.
     */
    template <typename T, class VariablesType>
    void fragmentation(vnegpu::graph<T,VariablesType>* base_graph, graph<T,VariablesType> *graph, int* final_result)
    {
      vnegpu::algorithm::detail::fragmentation_imp(base_graph, graph, final_result);
    }

    template <typename T, class VariablesType>
    void fingerprint(vnegpu::graph<T,VariablesType>* base_graph, graph<T,VariablesType> *graph, T* final_result)
    {
      vnegpu::algorithm::detail::fingerprint_imp(base_graph, graph, final_result);
    }

    template <typename T, class VariablesType>
    void percent_util(vnegpu::graph<T,VariablesType>* base_graph, graph<T,VariablesType> *graph, int* final_result)
    {
      vnegpu::algorithm::detail::percent_util_imp(base_graph, graph, final_result);
    }

    template <typename T, class VariablesType>
    T sum_variable_node(graph<T,VariablesType> *graph, int variable)
    {
      return vnegpu::algorithm::detail::sum_variable_node_imp(graph, variable);
    }

    template <typename T, class VariablesType>
    T sum_variable_edge(graph<T,VariablesType> *graph, int variable)
    {
      return vnegpu::algorithm::detail::sum_variable_edge_imp(graph, variable);
    }

    template <typename T, class VariablesType>
    T sum_variable_edge_used(graph<T,VariablesType> *graph, int variable){
      return vnegpu::algorithm::detail::sum_variable_edge_used_imp(graph, variable);
    }


  }//end algorithm
}//end vnegpu



#endif
