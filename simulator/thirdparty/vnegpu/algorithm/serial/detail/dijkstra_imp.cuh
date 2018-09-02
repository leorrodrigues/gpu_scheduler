#ifndef _SERIAL_DIJKSTRA_IMP_CUH
#define _SERIAL_DIJKSTRA_IMP_CUH

/*! \file
 *  \brief Serial Dijkstra Core/Implementation functions
 */
#include <limits>
#include <queue>
#include <vector>

#include <vnegpu/graph.cuh>

#include <vnegpu/util/util.cuh>

namespace vnegpu
{
  namespace algorithm
  {
    namespace serial
    {
      namespace detail
      {
        /**
         * \brief Calculate the distance starting from a node on the graph.
         * \param initial_node The initial node to calculate the distance.
         * \param graph_distance The graph to be used.
         * \param distance_index The edge variable index to save the metric.
         * \param metric_functor The functor to be used to calculate.
         */
        template <typename T, class VariablesType, class MetricFunctor>
        void dijkstra_imp(graph<T,VariablesType> *graph_distance, int initial_node, int distance_index, MetricFunctor metric_functor)
        {

          T infinite = std::numeric_limits<T>::max();

          std::pair<T, int> initial_pair(0.0f, initial_node);

          for(int y=0; y<graph_distance->get_num_nodes(); y++){
              graph_distance->set_variable_node(distance_index, y, infinite);
          }
          graph_distance->set_variable_node(distance_index, initial_node, 0);

          std::priority_queue<std::pair<T, int>, std::vector<std::pair<T, int>>, std::greater<std::pair<T, int>> > queue;
          queue.push(initial_pair);

          while(!queue.empty())
          {
            std::pair<T, int> element = queue.top();
            queue.pop();

            int self_distance = element.first;
            int node = element.second;

            if(self_distance > graph_distance->get_variable_node(distance_index, node))
                continue;

            for(int i=graph_distance->get_source_offset(node); i<graph_distance->get_source_offset(node+1); i++)
            {
              T distance = metric_functor(graph_distance, i, node);
              int neighbor = graph_distance->get_destination_indice(i);
              T new_distance = self_distance+distance;
              if(new_distance < graph_distance->get_variable_node(distance_index, neighbor)){
                  graph_distance->set_variable_node(distance_index, neighbor, new_distance);
                  std::pair<T, int> new_pair(new_distance, neighbor);
                  queue.push(new_pair);
              }
            }
          }

        }

      }//end detail
    }//end serial
  }//end algorithm
}//end vnegpu



#endif
