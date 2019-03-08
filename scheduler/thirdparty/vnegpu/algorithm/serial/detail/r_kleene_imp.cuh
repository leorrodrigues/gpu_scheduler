#ifndef _SERIAL_R_KLEENE_IMP_CUH
#define _SERIAL_R_KLEENE_IMP_CUH

/*! \file
 *  \brief Serial R-Kleene Core/Implementation functions
 */

#include <limits>

#include <vnegpu/graph.cuh>

#include <vnegpu/util/util.cuh>
#include <vnegpu/util/host_matrix.cuh>

namespace vnegpu
{
  namespace algorithm
  {
    namespace serial
    {
      namespace detail
      {

        /**
         * \brief Floyd Warshall algorithm.
         * \param distance_matrix The matrix to be used.
         */
        template <typename T>
        void floyd_warshall(vnegpu::util::host_matrix<T>* distance_matrix)
        {
            int num_elements = distance_matrix->num_colunas;

            for (int k = 0; k < num_elements; k++)
            {
              for (int i = 0; i < num_elements; i++)
              {
                for (int j = 0; j < num_elements; j++)
                {
                  T val_a = distance_matrix->get_element(i, k);
                  T val_b = distance_matrix->get_element(k, j);
                  distance_matrix->set_element(i, j, fminf(val_a + val_b, distance_matrix->get_element(i, j)));
                }
              }
            }
        }

        /**
         * \brief Calculate the modificated matrix multiplacation function for R-Kleene.
         * \param matrix_a The first matrix.
         * \param matrix_b The second matrix.
         * \param matrix_result The result matrix.
         * \param result_plus Control if the result should be calculated with the previuos matrix result value.
         * \param infinite The max distance that can be stored.
         */
        template <typename T>
        void matrix_mul_min(vnegpu::util::host_matrix<T>* matrix_a,
                            vnegpu::util::host_matrix<T>* matrix_b,
                            vnegpu::util::host_matrix<T>* matrix_result,
                            bool result_plus,
                            T infinite)
        {
          for (int i = 0; i < matrix_a->num_linhas; i++)
          {
            for (int j = 0; j < matrix_b->num_colunas; j++)
            {
              T register final_value = infinite;
              for (int k = 0; k < matrix_a->num_colunas; k++)
              {
                T val_a = matrix_a->get_element(i, k);
                T val_b = matrix_b->get_element(k, j);
                final_value = fminf(val_a + val_b, final_value);
              }

              if(result_plus){
                matrix_result->set_element(i, j, fminf(final_value, matrix_result->get_element(i, j)));
              }else{
                matrix_result->set_element(i, j, final_value);
              }

            }
          }
        }

        /**
         * \brief R-Kleene main loop.
         * \param distance_matrix The matrix to be saved.
         * \param infinite The max distance that can be stored.
         */
        template <typename T>
        void r_kleene_loop(vnegpu::util::host_matrix<T>* distance_matrix, T infinite)
        {
            if(distance_matrix->num_linhas < CUDA_BLOCK_SIZE_SHARED)
            {
              floyd_warshall(distance_matrix);
            }else{
              int new_size_row = distance_matrix->num_linhas/2;
              int new_size_col = distance_matrix->num_colunas/2;
              vnegpu::util::host_matrix<T>* A = distance_matrix->sub_matrix(0, 0, new_size_row, new_size_col);
              vnegpu::util::host_matrix<T>* B = distance_matrix->sub_matrix(0, new_size_col, new_size_row, distance_matrix->num_colunas-new_size_col);
              vnegpu::util::host_matrix<T>* C = distance_matrix->sub_matrix(new_size_row, 0, distance_matrix->num_linhas-new_size_row, new_size_col);
              vnegpu::util::host_matrix<T>* D = distance_matrix->sub_matrix(new_size_row, new_size_col, distance_matrix->num_linhas-new_size_row, distance_matrix->num_colunas-new_size_col);

              /*
              A=R-Kleene(A);
              B=A⊗B;
              C=C⊗A;
              D=D⊕(C⊗B);
              D=R-Kleene(D);
              B=B⊗D;
              C=D⊗C;
              A=A⊕(B⊗C);
               */

              r_kleene_loop(A, infinite);

              matrix_mul_min(A, B, B, false, infinite);

              matrix_mul_min(C, A, C, false, infinite);

              matrix_mul_min(C, B, D, true, infinite);

              r_kleene_loop(D, infinite);

              matrix_mul_min(B, D, B, false, infinite);

              matrix_mul_min(D, C, C, false, infinite);

              matrix_mul_min(B, C, A, true, infinite);

            }

        }

        /**
         * \brief Generate the matrix of distances to be used on the R-Kleene Algorithm.
         * \param graph_distance The graph to be used.
         * \param distance_index The index of the adge variable used as distance.
         * \param distance_matrix The matrix to be saved.
         * \param infinite The max distance that can be stored.
         */
        template <typename T, class VariablesType>
        void construct_distance_matrix(graph<T,VariablesType>* graph_distance,
                                       vnegpu::util::host_matrix<T>* distance_matrix,
                                       int distance_index,
                                       T infinite)
        {
          for(int id=0;id<graph_distance->get_num_nodes();id++){

             for(int i=0;i<graph_distance->get_num_nodes();i++){
               if(i==id){
                 distance_matrix->set_element(id,i,0);
               }else{
                 distance_matrix->set_element(id,i,infinite);
               }
             }

             for(int i=graph_distance->get_source_offset(id); i<graph_distance->get_source_offset(id+1); i++)
             {
               int destination = graph_distance->get_destination_indice(i);

               T distance = graph_distance->get_variable_edge(distance_index, i);
               distance_matrix->set_element(id, destination, distance);
             }
          }

        }

        /**
         * \brief Calculate the distance from all nodes to all nodes.
         * \param graph_distance The graph to be used.
         * \param distance_index The index of the adge variable used as distance.
         * \return The result distance matrix.
         */
        template <typename T, class VariablesType>
        vnegpu::util::host_matrix<T>* r_kleene_imp(graph<T,VariablesType> *graph_distance, int distance_index)
        {
            if(distance_index >= graph_distance->get_num_var_edges()){
              throw std::invalid_argument("The Distance Index is invalid.");
            }

            vnegpu::util::host_matrix<T>* distance_matrix = new vnegpu::util::host_matrix<T>(graph_distance->get_num_nodes(),graph_distance->get_num_nodes());

            T infinite = std::numeric_limits<T>::max();

            construct_distance_matrix(graph_distance, distance_matrix, distance_index, infinite);

            r_kleene_loop<T>(distance_matrix, infinite);

            return distance_matrix;

        }
      }//end detail
    }//end serial
  }//end algorithm
}//end vnegpu



#endif
