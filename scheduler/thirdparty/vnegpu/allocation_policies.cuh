#ifndef _ALOCATION_FUNCTORS_CUH
#define _ALOCATION_FUNCTORS_CUH

/*! \file
 *  \brief Alocaiion Functors
 */

#include <vnegpu/graph.cuh>
#include <vnegpu/metrics.cuh> //Pre-defined metrics
#include <vnegpu/algorithm/generic_rank.cuh>
#include <vnegpu/algorithm/local_metric.cuh> //Local metric algorithm

namespace vnegpu
{
  namespace allocation
  {
    /**
     * The distances are store on a matrix of size Nodes x Nodes.
     */

     struct worst_fit
     {

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_best_node(graph<T,VariablesType> *data_center, int a, int b)
       {
         return data_center->get_variable_node(data_center->variables.node_capacity, a) > data_center->get_variable_node(data_center->variables.node_capacity, b);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_node_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return ( data_center->get_node_type(host_id) == vnegpu::TYPE_HOST || data_center->get_node_type(host_id) == request->get_node_type(request_id) ) &&
         data_center->get_variable_node(data_center->variables.node_capacity, host_id) >= request->get_variable_node(request->variables.node_capacity, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_edge_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return data_center->get_variable_edge(data_center->variables.edge_capacity, host_id) >= request->get_variable_edge(request->variables.edge_capacity, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_capacity, host_id)-request->get_variable_edge(request->variables.edge_capacity, request_id);
         data_center->set_variable_edge(data_center->variables.edge_capacity, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_capacity, host_id)-request->get_variable_node(request->variables.node_capacity, request_id);
         data_center->set_variable_node(data_center->variables.node_capacity, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_capacity, host_id)+request->get_variable_edge(request->variables.edge_capacity, request_id);
         data_center->set_variable_edge(data_center->variables.edge_capacity, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_capacity, host_id)+request->get_variable_node(request->variables.node_capacity, request_id);
         data_center->set_variable_node(data_center->variables.node_capacity, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       T inline edge_distance(vnegpu::graph<T,VariablesType> *graph, int node1_id, int node2_id){
         return (T)1;
       }

       template <typename T, class VariablesType>
       __host__
       void inline node_each_iteration(vnegpu::graph<T,VariablesType> *data_center, vnegpu::graph<T,VariablesType> *request){

       }

     };


     struct worst_fit_pagerank
     {
       template <typename T, class VariablesType>
       __host__ __device__
       bool is_best_node(graph<T,VariablesType> *data_center, int a, int b)
       {
         return data_center->get_variable_node(data_center->variables.node_rank, a) > data_center->get_variable_node(data_center->variables.node_rank, b);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_node_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return ( data_center->get_node_type(host_id) == vnegpu::TYPE_HOST || data_center->get_node_type(host_id) == request->get_node_type(request_id) ) &&
         data_center->get_variable_node(data_center->variables.node_capacity, host_id) >= request->get_variable_node(request->variables.node_capacity, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_edge_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return data_center->get_variable_edge(data_center->variables.edge_capacity, host_id) >= request->get_variable_edge(request->variables.edge_capacity, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_capacity, host_id)-request->get_variable_edge(request->variables.edge_capacity, request_id);
         data_center->set_variable_edge(data_center->variables.edge_capacity, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_capacity, host_id)-request->get_variable_node(request->variables.node_capacity, request_id);
         data_center->set_variable_node(data_center->variables.node_capacity, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_capacity, host_id)+request->get_variable_edge(request->variables.edge_capacity, request_id);
         data_center->set_variable_edge(data_center->variables.edge_capacity, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_capacity, host_id)+request->get_variable_node(request->variables.node_capacity, request_id);
         data_center->set_variable_node(data_center->variables.node_capacity, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       T inline edge_distance(vnegpu::graph<T,VariablesType> *graph, int node1_id, int node2_id){
         return (T)1;
       }

       template <typename T, class VariablesType>
       __host__
       void inline node_each_iteration(vnegpu::graph<T,VariablesType> *data_center, vnegpu::graph<T,VariablesType> *request){
          vnegpu::algorithm::generic_rank(data_center, data_center->variables.node_rank, vnegpu::metrics::node_capacity() );
       }

     };


     struct worst_fit_group
     {
       template <typename T, class VariablesType>
       __host__ __device__
       bool is_best_node(graph<T,VariablesType> *data_center, int a, int b)
       {
         return data_center->get_variable_node(data_center->variables.node_capacity, a) > data_center->get_variable_node(data_center->variables.node_capacity, b);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_node_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return ( data_center->get_node_type(host_id) == vnegpu::TYPE_HOST || data_center->get_node_type(host_id) == request->get_node_type(request_id) ) &&
         data_center->get_group_id(host_id) == request->get_allocation_to_nodes_ids(request_id) &&
         data_center->get_variable_node(data_center->variables.node_capacity, host_id) >= request->get_variable_node(request->variables.node_capacity, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_edge_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return data_center->get_variable_edge(data_center->variables.edge_capacity, host_id) >= request->get_variable_edge(request->variables.edge_capacity, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_capacity, host_id)-request->get_variable_edge(request->variables.edge_capacity, request_id);
         data_center->set_variable_edge(data_center->variables.edge_capacity, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_capacity, host_id)-request->get_variable_node(request->variables.node_capacity, request_id);
         data_center->set_variable_node(data_center->variables.node_capacity, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_capacity, host_id)+request->get_variable_edge(request->variables.edge_capacity, request_id);
         data_center->set_variable_edge(data_center->variables.edge_capacity, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_capacity, host_id)+request->get_variable_node(request->variables.node_capacity, request_id);
         data_center->set_variable_node(data_center->variables.node_capacity, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       T inline edge_distance(vnegpu::graph<T,VariablesType> *graph, int node1_id, int node2_id){
         return (T)1;
       }

       template <typename T, class VariablesType>
       __host__
       void inline node_each_iteration(vnegpu::graph<T,VariablesType> *data_center, vnegpu::graph<T,VariablesType> *request){

       }
     };


       struct best_fit
       {
         template <typename T, class VariablesType>
         __host__ __device__
         bool is_best_node(graph<T,VariablesType> *data_center, int a, int b)
         {
           return data_center->get_variable_node(data_center->variables.node_capacity, a) < data_center->get_variable_node(data_center->variables.node_capacity, b);
         }

         template <typename T, class VariablesType>
         __host__ __device__
         bool is_node_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
         {
           return ( data_center->get_node_type(host_id) == vnegpu::TYPE_HOST || data_center->get_node_type(host_id) == request->get_node_type(request_id) ) &&
           data_center->get_variable_node(data_center->variables.node_capacity, host_id) >= request->get_variable_node(request->variables.node_capacity, request_id);
         }

         template <typename T, class VariablesType>
         __host__ __device__
         bool is_edge_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
         {
           return data_center->get_variable_edge(data_center->variables.edge_capacity, host_id) >= request->get_variable_edge(request->variables.edge_capacity, request_id);
         }

         template <typename T, class VariablesType>
         __host__ __device__
         void alloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
         {
           T new_value = data_center->get_variable_edge(data_center->variables.edge_capacity, host_id)-request->get_variable_edge(request->variables.edge_capacity, request_id);
           data_center->set_variable_edge(data_center->variables.edge_capacity, host_id, new_value);
         }

         template <typename T, class VariablesType>
         __host__ __device__
         void alloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
         {
           T new_value = data_center->get_variable_node(data_center->variables.node_capacity, host_id)-request->get_variable_node(request->variables.node_capacity, request_id);
           data_center->set_variable_node(data_center->variables.node_capacity, host_id, new_value);
         }

         template <typename T, class VariablesType>
         __host__ __device__
         void desalloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
         {
           T new_value = data_center->get_variable_edge(data_center->variables.edge_capacity, host_id)+request->get_variable_edge(request->variables.edge_capacity, request_id);
           data_center->set_variable_edge(data_center->variables.edge_capacity, host_id, new_value);
         }

         template <typename T, class VariablesType>
         __host__ __device__
         void desalloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
         {
           T new_value = data_center->get_variable_node(data_center->variables.node_capacity, host_id)+request->get_variable_node(request->variables.node_capacity, request_id);
           data_center->set_variable_node(data_center->variables.node_capacity, host_id, new_value);
         }

         template <typename T, class VariablesType>
         __host__ __device__
         T inline edge_distance(vnegpu::graph<T,VariablesType> *graph, int node1_id, int node2_id){
           return (T)1;
         }

         template <typename T, class VariablesType>
         __host__
         void inline node_each_iteration(vnegpu::graph<T,VariablesType> *data_center, vnegpu::graph<T,VariablesType> *request){

         }

       };


       struct best_fit_pagerank
       {
         template <typename T, class VariablesType>
         __host__ __device__
         bool is_best_node(graph<T,VariablesType> *data_center, int a, int b)
         {
           return data_center->get_variable_node(data_center->variables.node_rank, a) < data_center->get_variable_node(data_center->variables.node_rank, b);
         }

         template <typename T, class VariablesType>
         __host__ __device__
         bool is_node_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
         {
           return ( data_center->get_node_type(host_id) == vnegpu::TYPE_HOST || data_center->get_node_type(host_id) == request->get_node_type(request_id) ) &&
           data_center->get_variable_node(data_center->variables.node_capacity, host_id) >= request->get_variable_node(request->variables.node_capacity, request_id);
         }

         template <typename T, class VariablesType>
         __host__ __device__
         bool is_edge_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
         {
           return data_center->get_variable_edge(data_center->variables.edge_capacity, host_id) >= request->get_variable_edge(request->variables.edge_capacity, request_id);
         }

         template <typename T, class VariablesType>
         __host__ __device__
         void alloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
         {
           T new_value = data_center->get_variable_edge(data_center->variables.edge_capacity, host_id)-request->get_variable_edge(request->variables.edge_capacity, request_id);
           data_center->set_variable_edge(data_center->variables.edge_capacity, host_id, new_value);
         }

         template <typename T, class VariablesType>
         __host__ __device__
         void alloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
         {
           T new_value = data_center->get_variable_node(data_center->variables.node_capacity, host_id)-request->get_variable_node(request->variables.node_capacity, request_id);
           data_center->set_variable_node(data_center->variables.node_capacity, host_id, new_value);
         }

         template <typename T, class VariablesType>
         __host__ __device__
         void desalloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
         {
           T new_value = data_center->get_variable_edge(data_center->variables.edge_capacity, host_id)+request->get_variable_edge(request->variables.edge_capacity, request_id);
           data_center->set_variable_edge(data_center->variables.edge_capacity, host_id, new_value);
         }

         template <typename T, class VariablesType>
         __host__ __device__
         void desalloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
         {
           T new_value = data_center->get_variable_node(data_center->variables.node_capacity, host_id)+request->get_variable_node(request->variables.node_capacity, request_id);
           data_center->set_variable_node(data_center->variables.node_capacity, host_id, new_value);
         }

         template <typename T, class VariablesType>
         __host__ __device__
         T inline edge_distance(vnegpu::graph<T,VariablesType> *graph, int node1_id, int node2_id){
           return (T)1;
         }

         template <typename T, class VariablesType>
         __host__
         void inline node_each_iteration(vnegpu::graph<T,VariablesType> *data_center, vnegpu::graph<T,VariablesType> *request){
            vnegpu::algorithm::generic_rank(data_center, data_center->variables.node_rank, vnegpu::metrics::node_capacity() );
         }

       };

     struct best_fit_group
     {
       template <typename T, class VariablesType>
       __host__ __device__
       bool is_best_node(graph<T,VariablesType> *data_center, int a, int b)
       {
         return data_center->get_variable_node(data_center->variables.node_capacity, a) < data_center->get_variable_node(data_center->variables.node_capacity, b);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_node_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return ( data_center->get_node_type(host_id) == vnegpu::TYPE_HOST || data_center->get_node_type(host_id) == request->get_node_type(request_id) ) &&
         data_center->get_group_id(host_id) == request->get_allocation_to_nodes_ids(request_id) &&
         data_center->get_variable_node(data_center->variables.node_capacity, host_id) >= request->get_variable_node(request->variables.node_capacity, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_edge_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return data_center->get_variable_edge(data_center->variables.edge_capacity, host_id) >= request->get_variable_edge(request->variables.edge_capacity, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_capacity, host_id)-request->get_variable_edge(request->variables.edge_capacity, request_id);
         data_center->set_variable_edge(data_center->variables.edge_capacity, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_capacity, host_id)-request->get_variable_node(request->variables.node_capacity, request_id);
         data_center->set_variable_node(data_center->variables.node_capacity, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_capacity, host_id)+request->get_variable_edge(request->variables.edge_capacity, request_id);
         data_center->set_variable_edge(data_center->variables.edge_capacity, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_capacity, host_id)+request->get_variable_node(request->variables.node_capacity, request_id);
         data_center->set_variable_node(data_center->variables.node_capacity, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       T inline edge_distance(vnegpu::graph<T,VariablesType> *graph, int node1_id, int node2_id){
         return (T)1;
       }

       template <typename T, class VariablesType>
       __host__
       void inline node_each_iteration(vnegpu::graph<T,VariablesType> *data_center, vnegpu::graph<T,VariablesType> *request){

       }

     };




     struct worst_fit_basic_machine
     {
       template <typename T, class VariablesType>
       __host__ __device__
       bool is_best_node(graph<T,VariablesType> *data_center, int a, int b)
       {
         return data_center->get_variable_node(data_center->variables.node_cpu, a) > data_center->get_variable_node(data_center->variables.node_cpu, b);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_node_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return data_center->get_node_type(host_id) == vnegpu::TYPE_HOST &&
         data_center->get_variable_node(data_center->variables.node_cpu, host_id) >= request->get_variable_node(request->variables.node_cpu, request_id) &&
         data_center->get_variable_node(data_center->variables.node_memory, host_id) >= request->get_variable_node(request->variables.node_memory, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_edge_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return data_center->get_variable_edge(data_center->variables.edge_band, host_id) >= request->get_variable_edge(request->variables.edge_band, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_band, host_id)-request->get_variable_edge(request->variables.edge_band, request_id);
         data_center->set_variable_edge(data_center->variables.edge_band, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_cpu, host_id)-request->get_variable_node(request->variables.node_cpu, request_id);
         data_center->set_variable_node(data_center->variables.node_cpu, host_id, new_value);
         new_value = data_center->get_variable_node(data_center->variables.node_memory, host_id)-request->get_variable_node(request->variables.node_memory, request_id);
         data_center->set_variable_node(data_center->variables.node_memory, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_band, host_id)+request->get_variable_edge(request->variables.edge_band, request_id);
         data_center->set_variable_edge(data_center->variables.edge_band, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_cpu, host_id)+request->get_variable_node(request->variables.node_cpu, request_id);
         data_center->set_variable_node(data_center->variables.node_cpu, host_id, new_value);
         new_value = data_center->get_variable_node(data_center->variables.node_memory, host_id)+request->get_variable_node(request->variables.node_memory, request_id);
         data_center->set_variable_node(data_center->variables.node_memory, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       T inline edge_distance(vnegpu::graph<T,VariablesType> *graph, int node1_id, int node2_id){
         return (T)1;
       }

       template <typename T, class VariablesType>
       __host__
       void inline node_each_iteration(vnegpu::graph<T,VariablesType> *data_center, vnegpu::graph<T,VariablesType> *request){

       }

     };


     struct worst_fit_basic_machine_LRC
     {
       template <typename T, class VariablesType>
       __host__ __device__
       bool is_best_node(graph<T,VariablesType> *data_center, int a, int b)
       {
         return data_center->get_variable_node(data_center->variables.node_rank, a) > data_center->get_variable_node(data_center->variables.node_rank, b);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_node_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return data_center->get_node_type(host_id) == vnegpu::TYPE_HOST &&
         data_center->get_variable_node(data_center->variables.node_cpu, host_id) >= request->get_variable_node(request->variables.node_cpu, request_id) &&
         data_center->get_variable_node(data_center->variables.node_memory, host_id) >= request->get_variable_node(request->variables.node_memory, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_edge_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return data_center->get_variable_edge(data_center->variables.edge_band, host_id) >= request->get_variable_edge(request->variables.edge_band, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_band, host_id)-request->get_variable_edge(request->variables.edge_band, request_id);
         data_center->set_variable_edge(data_center->variables.edge_band, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_cpu, host_id)-request->get_variable_node(request->variables.node_cpu, request_id);
         data_center->set_variable_node(data_center->variables.node_cpu, host_id, new_value);
         new_value = data_center->get_variable_node(data_center->variables.node_memory, host_id)-request->get_variable_node(request->variables.node_memory, request_id);
         data_center->set_variable_node(data_center->variables.node_memory, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_band, host_id)+request->get_variable_edge(request->variables.edge_band, request_id);
         data_center->set_variable_edge(data_center->variables.edge_band, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_cpu, host_id)+request->get_variable_node(request->variables.node_cpu, request_id);
         data_center->set_variable_node(data_center->variables.node_cpu, host_id, new_value);
         new_value = data_center->get_variable_node(data_center->variables.node_memory, host_id)+request->get_variable_node(request->variables.node_memory, request_id);
         data_center->set_variable_node(data_center->variables.node_memory, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       T inline edge_distance(vnegpu::graph<T,VariablesType> *graph, int node1_id, int node2_id){
         return (T)1;
       }

       template <typename T, class VariablesType>
       __host__
       void inline node_each_iteration(vnegpu::graph<T,VariablesType> *data_center, vnegpu::graph<T,VariablesType> *request){
          vnegpu::algorithm::local_metric(data_center, data_center->variables.node_rank, vnegpu::metrics::LRC_machine() );
       }

     };


     struct best_fit_basic_machine_LRC
     {
       template <typename T, class VariablesType>
       __host__ __device__
       bool is_best_node(graph<T,VariablesType> *data_center, int a, int b)
       {
         return data_center->get_variable_node(data_center->variables.node_rank, a) < data_center->get_variable_node(data_center->variables.node_rank, b);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_node_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return data_center->get_node_type(host_id) == vnegpu::TYPE_HOST &&
         data_center->get_variable_node(data_center->variables.node_cpu, host_id) >= request->get_variable_node(request->variables.node_cpu, request_id) &&
         data_center->get_variable_node(data_center->variables.node_memory, host_id) >= request->get_variable_node(request->variables.node_memory, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_edge_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return data_center->get_variable_edge(data_center->variables.edge_band, host_id) >= request->get_variable_edge(request->variables.edge_band, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_band, host_id)-request->get_variable_edge(request->variables.edge_band, request_id);
         data_center->set_variable_edge(data_center->variables.edge_band, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_cpu, host_id)-request->get_variable_node(request->variables.node_cpu, request_id);
         data_center->set_variable_node(data_center->variables.node_cpu, host_id, new_value);
         new_value = data_center->get_variable_node(data_center->variables.node_memory, host_id)-request->get_variable_node(request->variables.node_memory, request_id);
         data_center->set_variable_node(data_center->variables.node_memory, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_band, host_id)+request->get_variable_edge(request->variables.edge_band, request_id);
         data_center->set_variable_edge(data_center->variables.edge_band, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_cpu, host_id)+request->get_variable_node(request->variables.node_cpu, request_id);
         data_center->set_variable_node(data_center->variables.node_cpu, host_id, new_value);
         new_value = data_center->get_variable_node(data_center->variables.node_memory, host_id)+request->get_variable_node(request->variables.node_memory, request_id);
         data_center->set_variable_node(data_center->variables.node_memory, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       T inline edge_distance(vnegpu::graph<T,VariablesType> *graph, int node1_id, int node2_id){
         return (T)1;
       }

       template <typename T, class VariablesType>
       __host__
       void inline node_each_iteration(vnegpu::graph<T,VariablesType> *data_center, vnegpu::graph<T,VariablesType> *request){
          vnegpu::algorithm::local_metric(data_center, data_center->variables.node_rank, vnegpu::metrics::LRC_machine() );
       }

     };


     struct worst_fit_basic_machine_pagerank
     {
       template <typename T, class VariablesType>
       __host__ __device__
       bool is_best_node(graph<T,VariablesType> *data_center, int a, int b)
       {
         return data_center->get_variable_node(data_center->variables.node_rank, a) > data_center->get_variable_node(data_center->variables.node_rank, b);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_node_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return data_center->get_node_type(host_id) == vnegpu::TYPE_HOST &&
         data_center->get_variable_node(data_center->variables.node_cpu, host_id) >= request->get_variable_node(request->variables.node_cpu, request_id) &&
         data_center->get_variable_node(data_center->variables.node_memory, host_id) >= request->get_variable_node(request->variables.node_memory, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_edge_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return data_center->get_variable_edge(data_center->variables.edge_band, host_id) >= request->get_variable_edge(request->variables.edge_band, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_band, host_id)-request->get_variable_edge(request->variables.edge_band, request_id);
         data_center->set_variable_edge(data_center->variables.edge_band, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_cpu, host_id)-request->get_variable_node(request->variables.node_cpu, request_id);
         data_center->set_variable_node(data_center->variables.node_cpu, host_id, new_value);
         new_value = data_center->get_variable_node(data_center->variables.node_memory, host_id)-request->get_variable_node(request->variables.node_memory, request_id);
         data_center->set_variable_node(data_center->variables.node_memory, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_band, host_id)+request->get_variable_edge(request->variables.edge_band, request_id);
         data_center->set_variable_edge(data_center->variables.edge_band, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_cpu, host_id)+request->get_variable_node(request->variables.node_cpu, request_id);
         data_center->set_variable_node(data_center->variables.node_cpu, host_id, new_value);
         new_value = data_center->get_variable_node(data_center->variables.node_memory, host_id)+request->get_variable_node(request->variables.node_memory, request_id);
         data_center->set_variable_node(data_center->variables.node_memory, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       T inline edge_distance(vnegpu::graph<T,VariablesType> *graph, int node1_id, int node2_id){
         return (T)1;
       }

       template <typename T, class VariablesType>
       __host__
       void inline node_each_iteration(vnegpu::graph<T,VariablesType> *data_center, vnegpu::graph<T,VariablesType> *request){
          vnegpu::algorithm::generic_rank(data_center, data_center->variables.node_rank, vnegpu::metrics::LRC_machine() );
       }

     };


     struct worst_fit_basic_machine_group
     {
       template <typename T, class VariablesType>
       __host__ __device__
       bool is_best_node(graph<T,VariablesType> *data_center, int a, int b)
       {
         return data_center->get_variable_node(data_center->variables.node_cpu, a) > data_center->get_variable_node(data_center->variables.node_cpu, b);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_node_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {

         return data_center->get_node_type(host_id) == vnegpu::TYPE_HOST &&
         data_center->get_group_id(host_id) == request->get_allocation_to_nodes_ids(request_id) &&
         data_center->get_variable_node(data_center->variables.node_cpu, host_id) >= request->get_variable_node(request->variables.node_cpu, request_id) &&
         data_center->get_variable_node(data_center->variables.node_memory, host_id) >= request->get_variable_node(request->variables.node_memory, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_edge_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return data_center->get_variable_edge(data_center->variables.edge_band, host_id) >= request->get_variable_edge(request->variables.edge_band, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_band, host_id)-request->get_variable_edge(request->variables.edge_band, request_id);
         data_center->set_variable_edge(data_center->variables.edge_band, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_cpu, host_id)-request->get_variable_node(request->variables.node_cpu, request_id);
         data_center->set_variable_node(data_center->variables.node_cpu, host_id, new_value);
         new_value = data_center->get_variable_node(data_center->variables.node_memory, host_id)-request->get_variable_node(request->variables.node_memory, request_id);
         data_center->set_variable_node(data_center->variables.node_memory, host_id, new_value);

       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_band, host_id)+request->get_variable_edge(request->variables.edge_band, request_id);
         data_center->set_variable_edge(data_center->variables.edge_band, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_cpu, host_id)+request->get_variable_node(request->variables.node_cpu, request_id);
         data_center->set_variable_node(data_center->variables.node_cpu, host_id, new_value);
         new_value = data_center->get_variable_node(data_center->variables.node_memory, host_id)+request->get_variable_node(request->variables.node_memory, request_id);
         data_center->set_variable_node(data_center->variables.node_memory, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       T inline edge_distance(vnegpu::graph<T,VariablesType> *graph, int node1_id, int node2_id){
         return (T)1;
       }

       template <typename T, class VariablesType>
       __host__
       void inline node_each_iteration(vnegpu::graph<T,VariablesType> *data_center, vnegpu::graph<T,VariablesType> *request){

       }

     };

     struct best_fit_basic_machine
     {
       template <typename T, class VariablesType>
       __host__ __device__
       bool is_best_node(graph<T,VariablesType> *data_center, int a, int b)
       {
         return data_center->get_variable_node(data_center->variables.node_cpu, a) < data_center->get_variable_node(data_center->variables.node_cpu, b);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_node_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return data_center->get_node_type(host_id) == vnegpu::TYPE_HOST &&
         data_center->get_variable_node(data_center->variables.node_cpu, host_id) >= request->get_variable_node(request->variables.node_cpu, request_id) &&
         data_center->get_variable_node(data_center->variables.node_memory, host_id) >= request->get_variable_node(request->variables.node_memory, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_edge_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return data_center->get_variable_edge(data_center->variables.edge_band, host_id) >= request->get_variable_edge(request->variables.edge_band, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_band, host_id)-request->get_variable_edge(request->variables.edge_band, request_id);
         data_center->set_variable_edge(data_center->variables.edge_band, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_cpu, host_id)-request->get_variable_node(request->variables.node_cpu, request_id);
         data_center->set_variable_node(data_center->variables.node_cpu, host_id, new_value);
         new_value = data_center->get_variable_node(data_center->variables.node_memory, host_id)-request->get_variable_node(request->variables.node_memory, request_id);
         data_center->set_variable_node(data_center->variables.node_memory, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_band, host_id)+request->get_variable_edge(request->variables.edge_band, request_id);
         data_center->set_variable_edge(data_center->variables.edge_band, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_cpu, host_id)+request->get_variable_node(request->variables.node_cpu, request_id);
         data_center->set_variable_node(data_center->variables.node_cpu, host_id, new_value);
         new_value = data_center->get_variable_node(data_center->variables.node_memory, host_id)+request->get_variable_node(request->variables.node_memory, request_id);
         data_center->set_variable_node(data_center->variables.node_memory, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       T inline edge_distance(vnegpu::graph<T,VariablesType> *graph, int node1_id, int node2_id){
         return (T)1;
       }

       template <typename T, class VariablesType>
       __host__
       void inline node_each_iteration(vnegpu::graph<T,VariablesType> *data_center, vnegpu::graph<T,VariablesType> *request){

       }

     };


     struct best_fit_basic_machine_max_check
     {
       template <typename T, class VariablesType>
       __host__ __device__
       bool is_best_node(graph<T,VariablesType> *data_center, int a, int b)
       {
         return data_center->get_variable_node(data_center->variables.node_cpu, a) < data_center->get_variable_node(data_center->variables.node_cpu, b);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_node_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return data_center->get_node_type(host_id) == vnegpu::TYPE_HOST &&
         data_center->get_variable_node(data_center->variables.node_cpu, host_id) >= request->get_variable_node(request->variables.node_cpu, request_id) &&
         data_center->get_variable_node(data_center->variables.node_memory, host_id) >= request->get_variable_node(request->variables.node_memory, request_id) &&
         data_center->get_variable_node(data_center->variables.node_rank, host_id) >= request->get_variable_node(request->variables.node_rank, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_edge_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return data_center->get_variable_edge(data_center->variables.edge_band, host_id) >= request->get_variable_edge(request->variables.edge_band, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_band, host_id)-request->get_variable_edge(request->variables.edge_band, request_id);
         data_center->set_variable_edge(data_center->variables.edge_band, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_cpu, host_id)-request->get_variable_node(request->variables.node_cpu, request_id);
         data_center->set_variable_node(data_center->variables.node_cpu, host_id, new_value);
         new_value = data_center->get_variable_node(data_center->variables.node_memory, host_id)-request->get_variable_node(request->variables.node_memory, request_id);
         data_center->set_variable_node(data_center->variables.node_memory, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_band, host_id)+request->get_variable_edge(request->variables.edge_band, request_id);
         data_center->set_variable_edge(data_center->variables.edge_band, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_cpu, host_id)+request->get_variable_node(request->variables.node_cpu, request_id);
         data_center->set_variable_node(data_center->variables.node_cpu, host_id, new_value);
         new_value = data_center->get_variable_node(data_center->variables.node_memory, host_id)+request->get_variable_node(request->variables.node_memory, request_id);
         data_center->set_variable_node(data_center->variables.node_memory, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       T inline edge_distance(vnegpu::graph<T,VariablesType> *graph, int node1_id, int node2_id){
         return (T)1;
       }

       template <typename T, class VariablesType>
       __host__
       void inline node_each_iteration(vnegpu::graph<T,VariablesType> *data_center, vnegpu::graph<T,VariablesType> *request){

       }

     };



     struct best_fit_basic_machine_pagerank
     {
       template <typename T, class VariablesType>
       __host__ __device__
       bool is_best_node(graph<T,VariablesType> *data_center, int a, int b)
       {
         return data_center->get_variable_node(data_center->variables.node_rank, a) < data_center->get_variable_node(data_center->variables.node_rank, b);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_node_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return data_center->get_node_type(host_id) == vnegpu::TYPE_HOST &&
         data_center->get_variable_node(data_center->variables.node_cpu, host_id) >= request->get_variable_node(request->variables.node_cpu, request_id) &&
         data_center->get_variable_node(data_center->variables.node_memory, host_id) >= request->get_variable_node(request->variables.node_memory, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_edge_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return data_center->get_variable_edge(data_center->variables.edge_band, host_id) >= request->get_variable_edge(request->variables.edge_band, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_band, host_id)-request->get_variable_edge(request->variables.edge_band, request_id);
         data_center->set_variable_edge(data_center->variables.edge_band, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_cpu, host_id)-request->get_variable_node(request->variables.node_cpu, request_id);
         data_center->set_variable_node(data_center->variables.node_cpu, host_id, new_value);
         new_value = data_center->get_variable_node(data_center->variables.node_memory, host_id)-request->get_variable_node(request->variables.node_memory, request_id);
         data_center->set_variable_node(data_center->variables.node_memory, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_band, host_id)+request->get_variable_edge(request->variables.edge_band, request_id);
         data_center->set_variable_edge(data_center->variables.edge_band, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_cpu, host_id)+request->get_variable_node(request->variables.node_cpu, request_id);
         data_center->set_variable_node(data_center->variables.node_cpu, host_id, new_value);
         new_value = data_center->get_variable_node(data_center->variables.node_memory, host_id)+request->get_variable_node(request->variables.node_memory, request_id);
         data_center->set_variable_node(data_center->variables.node_memory, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       T inline edge_distance(vnegpu::graph<T,VariablesType> *graph, int node1_id, int node2_id){
         return (T)1;
       }

       template <typename T, class VariablesType>
       __host__
       void inline node_each_iteration(vnegpu::graph<T,VariablesType> *data_center, vnegpu::graph<T,VariablesType> *request){
          vnegpu::algorithm::generic_rank(data_center, data_center->variables.node_rank, vnegpu::metrics::LRC_machine() );
       }

     };


     struct best_fit_basic_machine_group
     {
       template <typename T, class VariablesType>
       __host__ __device__
       bool is_best_node(graph<T,VariablesType> *data_center, int a, int b)
       {
         return data_center->get_variable_node(data_center->variables.node_cpu, a) < data_center->get_variable_node(data_center->variables.node_cpu, b);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_node_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return data_center->get_node_type(host_id) == vnegpu::TYPE_HOST &&
         data_center->get_group_id(host_id) == request->get_allocation_to_nodes_ids(request_id) &&
         data_center->get_variable_node(data_center->variables.node_cpu, host_id) >= request->get_variable_node(request->variables.node_cpu, request_id) &&
         data_center->get_variable_node(data_center->variables.node_memory, host_id) >= request->get_variable_node(request->variables.node_memory, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_edge_allocable(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         return data_center->get_variable_edge(data_center->variables.edge_band, host_id) >= request->get_variable_edge(request->variables.edge_band, request_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_band, host_id)-request->get_variable_edge(request->variables.edge_band, request_id);
         data_center->set_variable_edge(data_center->variables.edge_band, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_cpu, host_id)-request->get_variable_node(request->variables.node_cpu, request_id);
         data_center->set_variable_node(data_center->variables.node_cpu, host_id, new_value);
         new_value = data_center->get_variable_node(data_center->variables.node_memory, host_id)-request->get_variable_node(request->variables.node_memory, request_id);
         data_center->set_variable_node(data_center->variables.node_memory, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_edge(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_edge(data_center->variables.edge_band, host_id)+request->get_variable_edge(request->variables.edge_band, request_id);
         data_center->set_variable_edge(data_center->variables.edge_band, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_node(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, int host_id, int request_id)
       {
         T new_value = data_center->get_variable_node(data_center->variables.node_cpu, host_id)+request->get_variable_node(request->variables.node_cpu, request_id);
         data_center->set_variable_node(data_center->variables.node_cpu, host_id, new_value);
         new_value = data_center->get_variable_node(data_center->variables.node_memory, host_id)+request->get_variable_node(request->variables.node_memory, request_id);
         data_center->set_variable_node(data_center->variables.node_memory, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       T inline edge_distance(vnegpu::graph<T,VariablesType> *graph, int node1_id, int node2_id){
         return (T)1;
       }

       template <typename T, class VariablesType>
       __host__
       void inline node_each_iteration(vnegpu::graph<T,VariablesType> *data_center, vnegpu::graph<T,VariablesType> *request){

       }

     };

  }//end metrics
}//end vnegpu



#endif
