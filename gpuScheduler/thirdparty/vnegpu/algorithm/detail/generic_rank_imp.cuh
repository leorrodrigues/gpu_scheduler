#ifndef _GENERIC_RANK_IMP_CUH
#define _GENERIC_RANK_IMP_CUH

/*! \file
 *  \brief Generic Rank Implemented/Core functions
 */

#include <vnegpu/graph.cuh>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>

namespace vnegpu
{
  namespace algorithm
  {
    namespace detail
    {

      /**
       * \brief Initialize the rank required variables and do some pre-computation
       * \param rank_graph The graph to be used.
       * \param rank_index The node index to save the metric.
       * \param metric_functor The functor to be used to calculate.
       * \param d_temp_index Temporary array to store the neighbors metrics.
       */
      template <typename T, class VariablesType, class MetricFunctor>
      __global__ void
      initialize_rank_values(vnegpu::graph<T,VariablesType> rank_graph,
                             int rank_index,
                             MetricFunctor metric_functor,
                             T* d_temp_index)
      {
         int id = threadIdx.x + blockIdx.x * blockDim.x;

         if(id >= rank_graph.get_num_nodes())
              return;

         rank_graph.set_variable_node(rank_index, id, 1.0f / rank_graph.get_num_nodes() );

         float normalize_function = 0.0f;

         for(int i = rank_graph.get_source_offset(id); i < rank_graph.get_source_offset(id+1); i++)
         {
           normalize_function += metric_functor(&rank_graph,i,id);
         }

         d_temp_index[id]=normalize_function;

      }

      /**
       * \brief Calculate the R factor that is based on all nodes.
       * \param rank_graph The graph to be used.
       * \param rank_index The node index to save the metric.
       * \param p_factor The p factor from page rank.
       * \param d_r_index Temporary array to store R.
       */
      template <typename T, class VariablesType>
      __global__ void
      calculate_r_kernel(vnegpu::graph<T,VariablesType> rank_graph,
                         float p_factor,
                         int rank_index,
                         T* d_r_index)
      {
         int id = threadIdx.x + blockIdx.x * blockDim.x;

         if(id >= rank_graph.get_num_nodes())
              return;

         int outDegree = rank_graph.get_source_offset(id+1) - rank_graph.get_source_offset(id);

         T m = rank_graph.get_variable_node(rank_index,id) / (T)rank_graph.get_num_nodes();

         if(outDegree>0)
         {
            atomicAdd(d_r_index, (1.0 - p_factor) * m );
         }
         else
         {
            atomicAdd(d_r_index, m );
         }
      }

      /**
       * \brief Calculate the rank value of this iteration
       * \param rank_graph The graph to be used.
       * \param rank_index The node index to save the metric.
       * \param metric_functor The functor to be used to calculate.
       * \param d_temp_rank Temporary array to store the rank.
       * \param p_factor The p factor from pagerank.
       * \param r The R value for this iteration.
       * \param d_temp_index Temporary array to store the neighbors metrics.
       */
      template <typename T, class VariablesType, class MetricFunctor>
      __global__ void
      calculate_node_value(vnegpu::graph<T,VariablesType> rank_graph,
                           float p_factor,
                           int rank_index,
                           MetricFunctor metric_functor,
                           T* d_temp_rank,
                           T* d_temp_index,
                           T* r)
      {
        int id = threadIdx.x + blockIdx.x * blockDim.x;

        if(id >= rank_graph.get_num_nodes())
             return;

        register T res = *r;

        for(int i=rank_graph.get_source_offset(id); i<rank_graph.get_source_offset(id+1); i++)
        {
          T rank_link_weight = metric_functor(&rank_graph,i,id) / d_temp_index[ rank_graph.get_destination_indice(i) ];

          res += p_factor * rank_graph.get_variable_node(rank_index, rank_graph.get_destination_indice(i)) * rank_link_weight;
        }

        d_temp_rank[id] = res;
      }

      /**
       * \brief Calculate the rank value of this iteration
       * \param rank_graph The graph to be used.
       * \param rank_index The node index to save the metric.
       * \param d_temp_rank Temporary array to store the rank.
       * \param d_r_index Temporary array to store R.
       * \param error_factor The max error allowed to stop iterations.
       */
      template <typename T, class VariablesType>
      __global__ void
      update_node_value(vnegpu::graph<T,VariablesType> rank_graph,
                        int rank_index,
                        T* d_temp_rank,
                        bool* d_run,
                        T error_factor)
      {
        int id = threadIdx.x + blockIdx.x * blockDim.x;

        if(id >= rank_graph.get_num_nodes())
             return;

        if( (d_temp_rank[id]-rank_graph.get_variable_node(rank_index,id))/rank_graph.get_variable_node(rank_index,id) >= error_factor ){
           *d_run = true;
        }

        rank_graph.set_variable_node(rank_index, id, d_temp_rank[id] );
      }

      /**
       * \brief Apply the Generic Rank on the graph.
       * \param rank_graph The graph to be used.
       * \param rank_index The node index to save the rank.
       * \param metric_functor The functor to be used to calculate.
       * \param error_factor The error used to stop iterations.
       * \param p_factor The PageRank P factor.
       */
      template <typename T, class VariablesType, class MetricFunctor>
      void generic_rank_imp(vnegpu::graph<T,VariablesType> *rank_graph,
                            int rank_index,
                            MetricFunctor metric_functor,
                            float p_factor,
                            T error_factor)
      {

        //Checking if graph is on GPU memory.
        if(rank_graph->get_state()!=vnegpu::GRAPH_ON_GPU)
        {
          throw std::runtime_error("Graph is not on GPU.");
        }

        //Checking if rank_index is valid.
        if(rank_index >= rank_graph->get_num_var_nodes()){
          throw std::invalid_argument("The Rank Index is invalid.");
        }

        int num = rank_graph->get_num_nodes()/CUDA_BLOCK_SIZE + 1;
        dim3 Block(CUDA_BLOCK_SIZE);
        dim3 Grid(num);

        //temp arrays
        T* d_temp_index;
        cudaMalloc(&d_temp_index, sizeof(T) * rank_graph->get_num_nodes() );
        CUDA_CHECK();

        T* d_temp_rank;
        cudaMalloc(&d_temp_rank, sizeof(T) * rank_graph->get_num_nodes() );
        CUDA_CHECK();

        initialize_rank_values<<<Grid, Block>>>(*rank_graph,
                                                rank_index,
                                                metric_functor,
                                                d_temp_index);
        CUDA_CHECK();

        bool run = true;

        bool* d_run;
        cudaMalloc(&d_run, sizeof(bool));

        T* d_r_value;
        cudaMalloc(&d_r_value, sizeof(T));

        int ite = ITERATION_LIMIT;

        while(run && ite-->0)
        {
            //printf("IT:%d\n",ite);
            cudaMemset(d_run, 0, sizeof(bool));
            cudaMemset(d_r_value, 0, sizeof(T));

            calculate_r_kernel<<<Grid, Block>>>(*rank_graph,
                                                p_factor,
                                                rank_index,
                                                d_r_value);
            CUDA_CHECK();

            calculate_node_value<<<Grid, Block>>>(*rank_graph,
                                                  p_factor,
                                                  rank_index,
                                                  metric_functor,
                                                  d_temp_rank,
                                                  d_temp_index,
                                                  d_r_value);
            CUDA_CHECK();

            update_node_value<<<Grid, Block>>>(*rank_graph,
                                               rank_index,
                                               d_temp_rank,
                                               d_run,
                                               error_factor);
            CUDA_CHECK();

            cudaMemcpy(&run,d_run,sizeof(bool),cudaMemcpyDeviceToHost);

        }

        cudaFree(d_temp_index);
        cudaFree(d_temp_rank);
        cudaFree(d_run);
        cudaFree(d_r_value);

        cudaDeviceSynchronize();
      }
    }
  }
}

#endif
