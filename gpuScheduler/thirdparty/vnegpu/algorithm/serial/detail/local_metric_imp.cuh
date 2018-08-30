#ifndef _SERIAL_LOCAL_METRICS_IMP_CUH
#define _SERIAL_LOCAL_METRICS_IMP_CUH

/*! \file
 *  \brief Serial Local Metrics Core/Implementation functions
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
         * \brief Apply the local metric on the graph.
         * \param graph_metric The graph to be used.
         * \param rank_index The node index to save the metric.
         * \param metric_functor The functor to be used to calculate.
         */
        template <typename T, class VariablesType, class MetricFunctor>
        void local_metric_imp(graph<T,VariablesType> *graph_metric, int rank_index, MetricFunctor metric_functor)
        {
          if(rank_index >= graph_metric->get_num_var_nodes()){
            throw std::invalid_argument("The Rank Index is invalid.");
          }

          for(int id=0; id<graph_metric->get_num_nodes(); id++){
              graph_metric->set_variable_node(rank_index, id, metric_functor(graph_metric, 0, id) );
          }

        }
      }//end detail
    }//end serial
  }//end algorithm
}//end vnegpu

#endif
