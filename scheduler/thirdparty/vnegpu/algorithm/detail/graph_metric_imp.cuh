#ifndef _GRAPH_METRICS_IMP_CUH
#define _GRAPH_METRICS_IMP_CUH

/*! \file
 *  \brief Graph Metrics Core/Implementation functions
 */

#include <vnegpu/graph.cuh>

namespace vnegpu
{
  namespace algorithm
  {
    namespace detail
    {

      /**
       * \brief Kernel for local metric.
       * \param graph_metric The graph to be used.
       * \param rank_index The node index to save the metric.
       * \param metric_functor The functor to be used to calculate.
       */
      template <typename T, class VariablesType>
      __global__ void percent_util_kernel(vnegpu::graph<T,VariablesType> base_graph, vnegpu::graph<T,VariablesType> graph, int* result)
      {
        int id = threadIdx.x + blockIdx.x * blockDim.x;

        if(id < graph.get_num_nodes() && graph.get_node_type(id)==vnegpu::TYPE_HOST){
          float percent = (base_graph.get_variable_node(graph.variables.node_cpu, id)-graph.get_variable_node(graph.variables.node_cpu, id))/base_graph.get_variable_node(graph.variables.node_cpu, id);
          if(percent == 0.0){
              atomicAdd(&result[0], 1);
          }else if( percent < 0.2  ){
              atomicAdd(&result[1], 1);
          }else if(percent < 0.4){
              atomicAdd(&result[2], 1);
          }else if(percent < 0.6){
              atomicAdd(&result[3], 1);
          }else if(percent < 0.8){
              atomicAdd(&result[4], 1);
          }else{
              atomicAdd(&result[5], 1);
          }
        }

        if(id < graph.get_num_edges()){
          float percent = (base_graph.get_variable_edge_undirected(graph.variables.edge_band, id)-graph.get_variable_edge_undirected(graph.variables.edge_band, id))/base_graph.get_variable_edge_undirected(graph.variables.edge_band, id);
          if( percent == 0.0){
              atomicAdd(&result[6], 1);
          }else if(percent < 0.2){
              atomicAdd(&result[7], 1);
          }else if(percent < 0.4){
              atomicAdd(&result[8], 1);
          }else if(percent < 0.6){
              atomicAdd(&result[9], 1);
          }else if(percent < 0.8){
              atomicAdd(&result[10], 1);
          }else{
              atomicAdd(&result[11], 1);
          }
        }

      }

      /**
       * \brief Apply the local metric on the graph.
       * \param graph_metric The graph to be used.
       * \param rank_index The node index to save the metric.
       * \param metric_functor The functor to be used to calculate.
       */
      template <typename T, class VariablesType>
      void percent_util_imp(vnegpu::graph<T,VariablesType>* base_graph, graph<T,VariablesType> *graph, int* final_result)
      {

        final_result[0]=0;
        final_result[1]=0;
        final_result[2]=0;
        final_result[3]=0;
        final_result[4]=0;
        final_result[5]=0;
        final_result[6]=0;
        final_result[7]=0;
        final_result[8]=0;
        final_result[9]=0;
        final_result[10]=0;
        final_result[11]=0;

        int* d_result;
        cudaMalloc(&d_result, sizeof(int)*12);
        cudaMemset(d_result, 0, sizeof(int)*12);

        int thr = max(graph->get_num_nodes(), graph->get_num_edges());

        int num = thr/CUDA_BLOCK_SIZE + 1;
        dim3 Block(CUDA_BLOCK_SIZE);
        dim3 Grid(num);

        //Call the kernel
        percent_util_kernel<<<Grid, Block>>>(*base_graph, *graph, d_result); CUDA_CHECK();

        //Sync before the function end.
        cudaDeviceSynchronize();


        cudaMemcpy(final_result, d_result, sizeof(int)*12, cudaMemcpyDeviceToHost);
        cudaFree(d_result);

      }


      /**
       * \brief Kernel for local metric.
       * \param graph_metric The graph to be used.
       * \param rank_index The node index to save the metric.
       * \param metric_functor The functor to be used to calculate.
       */
      template <typename T, class VariablesType>
      __global__ void fragmentation_kernel(vnegpu::graph<T,VariablesType> base_graph, vnegpu::graph<T,VariablesType> graph, int* result)
      {
        int id = threadIdx.x + blockIdx.x * blockDim.x;

        if(id < graph.get_num_nodes()){
          if( base_graph.get_variable_node(graph.variables.node_cpu, id) != graph.get_variable_node(graph.variables.node_cpu, id) ){
              atomicAdd(&result[0], 1);
          }
        }

        if(id < graph.get_num_edges()){
          if(base_graph.get_variable_edge_undirected(graph.variables.edge_band, id) != graph.get_variable_edge_undirected(graph.variables.edge_band, id) ){
              atomicAdd(&result[1], 1);
          }
        }

      }

      /**
       * \brief Apply the local metric on the graph.
       * \param graph_metric The graph to be used.
       * \param rank_index The node index to save the metric.
       * \param metric_functor The functor to be used to calculate.
       */
      template <typename T, class VariablesType>
      void fragmentation_imp(vnegpu::graph<T,VariablesType>* base_graph, graph<T,VariablesType> *graph, int* final_result)
      {

        final_result[0]=0;
        final_result[1]=0;

        int* d_result;
        cudaMalloc(&d_result, sizeof(int)*2);
        cudaMemset(d_result, 0, sizeof(int)*2);

        int thr = max(graph->get_num_nodes(), graph->get_num_edges());

        int num = thr/CUDA_BLOCK_SIZE + 1;
        dim3 Block(CUDA_BLOCK_SIZE);
        dim3 Grid(num);

        //Call the kernel
        fragmentation_kernel<<<Grid, Block>>>(*base_graph, *graph, d_result); CUDA_CHECK();

        //Sync before the function end.
        cudaDeviceSynchronize();


        cudaMemcpy(final_result, d_result, sizeof(int)*2, cudaMemcpyDeviceToHost);
        cudaFree(d_result);
      }



      template <typename T, class VariablesType>
      __global__ void fingerprint_kernel(vnegpu::graph<T,VariablesType> base_graph, vnegpu::graph<T,VariablesType> graph, T* result)
      {
        int id = threadIdx.x + blockIdx.x * blockDim.x;

        if(id < graph.get_num_nodes()){
            atomicAdd(&result[0], graph.get_variable_node(graph.variables.node_cpu, id));
            atomicAdd(&result[1], graph.get_variable_node(graph.variables.node_memory, id));
        }

        if(id < graph.get_num_edges()){
            atomicAdd(&result[2], graph.get_variable_edge_undirected(graph.variables.edge_band, id));
        }

      }

      /**
       * \brief Apply the local metric on the graph.
       * \param graph_metric The graph to be used.
       * \param rank_index The node index to save the metric.
       * \param metric_functor The functor to be used to calculate.
       */
      template <typename T, class VariablesType>
      void fingerprint_imp(vnegpu::graph<T,VariablesType>* base_graph, graph<T,VariablesType> *graph, T* final_result)
      {

        final_result[0]=0;
        final_result[1]=0;
        final_result[2]=0;

        T* d_result;
        cudaMalloc(&d_result, sizeof(T)*3);
        cudaMemset(d_result, 0, sizeof(T)*3);

        int thr = max(graph->get_num_nodes(), graph->get_num_edges());

        int num = thr/CUDA_BLOCK_SIZE + 1;
        dim3 Block(CUDA_BLOCK_SIZE);
        dim3 Grid(num);

        //Call the kernel
        fingerprint_kernel<<<Grid, Block>>>(*base_graph, *graph, d_result); CUDA_CHECK();

        //Sync before the function end.
        cudaDeviceSynchronize();


        cudaMemcpy(final_result, d_result, sizeof(T)*3, cudaMemcpyDeviceToHost);
        cudaFree(d_result);
      }

      template <typename T, class VariablesType>
      T sum_variable_node_imp(graph<T,VariablesType> *graph, int variable)
      {
        T sum = 0;
        for(int i=0; i<graph->get_num_nodes(); i++){
          sum+=graph->get_variable_node(variable, i);
        }
        return sum;
      }

      template <typename T, class VariablesType>
      T sum_variable_edge_imp(graph<T,VariablesType> *graph, int variable)
      {
        T sum = 0;
        for(int i=0; i<graph->get_num_edges(); i++){
          sum+=graph->get_variable_edge(variable, i);
        }
        return sum;
      }

      template <typename T, class VariablesType>
      T sum_variable_edge_used_imp(graph<T,VariablesType> *graph, int variable)
      {
        T sum = 0;
        for(int i=0; i<graph->get_num_edges(); i++){
          int size = graph->get_allocation_to_edges_ids(i)->size();
          sum+=graph->get_variable_edge(variable, i)*size;
        }
        return sum;
      }
    }//end detail
  }//end algorithm
}//end vnegpu

#endif
