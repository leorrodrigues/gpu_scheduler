#ifndef _METRICS_CUH
#define _METRICS_CUH

/*! \file
 *  \brief Metrics Functors
 */

#include <vnegpu/graph.cuh>
#include <cmath>

namespace vnegpu
{
  namespace metrics
  {
    /**
     * Return one.
     */
    struct one
    {
      template <typename T, class VariablesType>
      __host__ __device__ T inline
      operator()(vnegpu::graph<T,VariablesType> *graph, int edge_id, int node_id){
        return (T)1;
      }
    };// end degree

    /**
     * Return Degree of the node.
     */
    struct degree
    {
      template <typename T, class VariablesType>
      __host__ __device__ T inline
      operator()(vnegpu::graph<T,VariablesType> *graph, int edge_id, int node_id){
        return (T)( graph->get_source_offset(node_id+1) - graph->get_source_offset(node_id) );
      }
    };// end degree

    /**
     * Return Degree of the node.
     */
    struct node_capacity
    {
      template <typename T, class VariablesType>
      __host__ __device__ T inline
      operator()(vnegpu::graph<T,VariablesType> *graph, int edge_id, int node_id){
        return graph->get_variable_node(graph->variables.node_capacity, node_id);
      }
    };// end degree

    struct node_cpu
    {
      template <typename T, class VariablesType>
      __host__ __device__ T inline
      operator()(vnegpu::graph<T,VariablesType> *graph, int edge_id, int node_id){
        return graph->get_variable_node(graph->variables.node_cpu, node_id);
      }
    };// end degree

    /**
     * Return the edge weight of the edge.
     */
    struct edge_weight
    {
      template <typename T, class VariablesType>
      __host__ __device__ T inline
      operator()(vnegpu::graph<T,VariablesType> *graph, int edge_id, int node_id){
        return graph->get_variable_edge(graph->variables.edge_capacity,edge_id);
      }
    };// end metric_edge_weight

    struct LRC_machine
    {
      template <typename T, class VariablesType>
      __host__ __device__ T inline
      operator()(vnegpu::graph<T,VariablesType> *graph, int edge_id, int node_id){
        T sum = 0.0;
        for(int i=graph->get_source_offset(node_id); i<graph->get_source_offset(node_id+1); i++)
        {
          sum+=graph->get_variable_edge(graph->variables.edge_band, i);
        }
        return graph->get_variable_node(graph->variables.node_cpu, node_id)*sum;
      }
    };// end metric_edge_weight

    struct probabilidade_falha
    {
      template <typename T, class VariablesType>
      __host__ __device__ T inline
      operator()(vnegpu::graph<T,VariablesType> *graph, int edge_id, int node_id){
        return floor(graph->get_variable_node(4, node_id)*1000000)/1000000;
      }
    };// end degree

  }//end metrics
}//end vnegpu



#endif
