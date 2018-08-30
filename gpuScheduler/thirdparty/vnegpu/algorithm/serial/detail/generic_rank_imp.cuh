#ifndef _SERIAL_GENERIC_RANK_IMP_CUH
#define _SERIAL_GENERIC_RANK_IMP_CUH

/*! \file
 *  \brief Serial Generic Rank Implemented/Core functions
 */

#include <vnegpu/graph.cuh>

namespace vnegpu
{
  namespace algorithm
  {
    namespace serial
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
        void
        initialize_rank_values_serial(vnegpu::graph<T,VariablesType>* rank_graph,
                               int rank_index,
                               MetricFunctor metric_functor,
                               T* d_temp_index)
        {
           for(int id=0;id<rank_graph->get_num_nodes();id++){

             rank_graph->set_variable_node(rank_index, id, 1.0f / rank_graph->get_num_nodes() );

             float normalize_function = 0.0f;

             for(int i = rank_graph->get_source_offset(id); i < rank_graph->get_source_offset(id+1); i++)
             {
               normalize_function += metric_functor(rank_graph,i,id);
             }

             d_temp_index[id]=normalize_function;

          }

        }

        /**
         * \brief Calculate the R factor that is based on all nodes.
         * \param rank_graph The graph to be used.
         * \param rank_index The node index to save the metric.
         * \param p_factor The p factor from page rank.
         * \param d_r_index Temporary array to store R.
         */
        template <typename T, class VariablesType>
        void
        calculate_r_kernel_serial(vnegpu::graph<T,VariablesType>* rank_graph,
                           float p_factor,
                           int rank_index,
                           T* d_r_index)
        {
           for(int id=0;id<rank_graph->get_num_nodes();id++){
             int outDegree = rank_graph->get_source_offset(id+1) - rank_graph->get_source_offset(id);

             T m = rank_graph->get_variable_node(rank_index,id) / (T)rank_graph->get_num_nodes();

             if(outDegree>0)
             {
                d_r_index[id]=(1.0 - p_factor) * m;
             }
             else
             {
                d_r_index[id]=m;
             }
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
        void
        calculate_node_value_serial(vnegpu::graph<T,VariablesType>* rank_graph,
                             float p_factor,
                             int rank_index,
                             MetricFunctor metric_functor,
                             T* d_temp_rank,
                             T* d_temp_index,
                             T r)
        {
          for(int id=0;id<rank_graph->get_num_nodes();id++){
            register T res = r;

            for(int i=rank_graph->get_source_offset(id); i<rank_graph->get_source_offset(id+1); i++)
            {
              T rank_link_weight = metric_functor(rank_graph,i,id) / d_temp_index[ rank_graph->get_destination_indice(i) ];

              res += p_factor * rank_graph->get_variable_node(rank_index, rank_graph->get_destination_indice(i)) * rank_link_weight;
            }

            d_temp_rank[id] = res;
          }
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
        void
        update_node_value_serial(vnegpu::graph<T,VariablesType>* rank_graph,
                          int rank_index,
                          T* d_temp_rank,
                          bool* run,
                          T error_factor)
        {
          for(int id=0;id<rank_graph->get_num_nodes();id++){

              if( (d_temp_rank[id]-rank_graph->get_variable_node(rank_index,id))/rank_graph->get_variable_node(rank_index,id) >= error_factor ){
                 *run=true;
              }

              rank_graph->set_variable_node(rank_index, id, d_temp_rank[id] );
          }
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
          T* d_temp_index = (T*)malloc(sizeof(T) * rank_graph->get_num_nodes() );

          T* d_r_index = (T*)malloc(sizeof(T) * rank_graph->get_num_nodes() );

          T* d_temp_rank = (T*)malloc(sizeof(T) * rank_graph->get_num_nodes() );

          initialize_rank_values_serial(rank_graph, rank_index, metric_functor, d_temp_index);

          bool conver = true;
          int ite = ITERATION_LIMIT;

          while(conver && ite-->0)
          {

              //printf("IT:%d\n",ite);
              conver = false;
              calculate_r_kernel_serial(rank_graph, p_factor, rank_index, d_r_index);

              T r = 0;

              for(int y=0; y<rank_graph->get_num_nodes(); y++){
                r += d_r_index[y];
              }

              calculate_node_value_serial(rank_graph,
                                                    p_factor,
                                                    rank_index,
                                                    metric_functor,
                                                    d_temp_rank,
                                                    d_temp_index,
                                                    r);

              update_node_value_serial(rank_graph,
                                                 rank_index,
                                                 d_temp_rank,
                                                 &conver,
                                                 error_factor);


          }

          free(d_temp_index);
          free(d_r_index);
          free(d_temp_rank);

        }
      }
    }// end serial
  }
}

#endif
