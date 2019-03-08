#ifndef _DISTANCE_CUH
#define _DISTANCE_CUH

/*! \file
 *  \brief Distance Functors
 */

#include <vnegpu/graph.cuh>

#include <vnegpu/util/matrix.cuh>
#include <vnegpu/util/host_matrix.cuh>

namespace vnegpu
{
  namespace distance
  {
    /**
     * The distances are store on a matrix of size Nodes x Nodes.
     */
    struct matrix_based
    {
      template <typename T>
      __device__ T inline operator()(vnegpu::util::matrix<T>* distance_matrix, int node1_id, int node2_id){
        return distance_matrix->get_element(node1_id, node2_id);
      }
    };// end degree

    namespace serial
    {
      struct matrix_based
      {
        template <typename T>
        __host__ T inline operator()(vnegpu::util::host_matrix<T>* distance_matrix, int node1_id, int node2_id){
          return distance_matrix->get_element(node1_id, node2_id);
        }
      };// end degree
    }

  }//end metrics
}//end vnegpu



#endif
