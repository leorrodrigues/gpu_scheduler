#ifndef _SERIAL_LOCAL_METRICS_CUH
#define _SERIAL_LOCAL_METRICS_CUH

/*! \file
 *  \brief Serial Local Metrics Interface functions
 */

#include <vnegpu/graph.cuh>
#include <vnegpu/algorithm/serial/detail/local_metric_imp.cuh>

namespace vnegpu
{
  namespace algorithm
  {
    namespace serial
    {
      /**
       * \brief Apply the local metric on the graph.
       * \param graph_rank The graph to be used.
       * \param rank_index The node index to save the metric.
       * \param metric_functor The functor to be used to calculate.
       */
      template <typename T, class VariablesType, class MetricFunctor>
      void local_metric(graph<T,VariablesType> *graph_metric, int rank_index, MetricFunctor metric_functor)
      {
        vnegpu::algorithm::serial::detail::local_metric_imp(graph_metric, rank_index, metric_functor);
      }
    }//end serial
  }//end algorithm
}//end vnegpu



#endif
