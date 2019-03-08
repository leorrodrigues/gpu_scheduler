#ifndef _R_KLEENE_IMP_CUH
#define _R_KLEENE_IMP_CUH

/*! \file
 *  \brief R-Kleene Core/Implementation functions
 */

#include <limits>

#include <vnegpu/graph.cuh>

#include <vnegpu/util/util.cuh>
#include <vnegpu/util/matrix.cuh>

namespace vnegpu
{
  namespace algorithm
  {
    namespace detail
    {

      /**
       * \brief Kernel that calculate the ASPS of a small matrix.
       * \param distance_matrix The matrix to be used.
       */
      template <typename T>
      __global__ void
      apsp_small_matrix(vnegpu::util::matrix<T> distance_matrix)
      {

          int tx = threadIdx.x;
          int ty = threadIdx.y;

          for (int k = 0; k < distance_matrix.num_colunas; ++k)
          {
            float M1 = distance_matrix.get_element(ty,k);
            float M2 = distance_matrix.get_element(k,tx);

            distance_matrix.set_element(ty, tx, fminf(M1+M2, distance_matrix.get_element(ty,tx)));

            __syncthreads();
          }
      }


      /**
       * \brief Kernel that calculate the modificated matrix multiplacation function for R-Kleene.
       * \param matrix_a The first matrix.
       * \param matrix_b The second matrix.
       * \param matrix_result The result matrix.
       * \param result_plus Control if the result should be calculated with the previuos matrix result value.
       * \param infinite The max distance that can be stored.
       */
      template <typename T>
      __global__ void
      matrix_mul_min_kernel(vnegpu::util::matrix<T> matrix_a,
                            vnegpu::util::matrix<T> matrix_b,
                            vnegpu::util::matrix<T> matrix_result,
                            bool result_plus,
                            T infinite)
      {

         int blockRow = blockIdx.y;
         int blockCol = blockIdx.x;

         T Cvalue = infinite;

         int tx = threadIdx.x;
         int ty = threadIdx.y;

         int Row = blockRow * CUDA_BLOCK_SIZE_SHARED + ty;
         int Col = blockCol * CUDA_BLOCK_SIZE_SHARED + tx;

         __shared__ T As[CUDA_BLOCK_SIZE_SHARED][CUDA_BLOCK_SIZE_SHARED];
         __shared__ T Bs[CUDA_BLOCK_SIZE_SHARED][CUDA_BLOCK_SIZE_SHARED];


         for (int m = 0; m < ( (matrix_a.num_colunas-1) / CUDA_BLOCK_SIZE_SHARED + 1); ++m) {

            if (Row < matrix_a.num_linhas && m*CUDA_BLOCK_SIZE_SHARED+tx < matrix_a.num_colunas){
                As[ty][tx] = matrix_a.get_element(Row, + m*CUDA_BLOCK_SIZE_SHARED+tx);
            }else{
                As[ty][tx] = infinite;
            }

            if (Col < matrix_b.num_colunas && m*CUDA_BLOCK_SIZE_SHARED+ty < matrix_b.num_linhas){
               Bs[ty][tx] = matrix_b.get_element(m*CUDA_BLOCK_SIZE_SHARED+ty,Col);
            }else{
               Bs[ty][tx] = infinite;
            }

             __syncthreads();

             #pragma unroll
             for (int e = 0; e < CUDA_BLOCK_SIZE_SHARED; ++e){
                 Cvalue = cuda_min<T>(As[ty][e] + Bs[e][tx], Cvalue);
              }

             __syncthreads();
         }

         if (Row < matrix_result.num_linhas && Col < matrix_result.num_colunas){
            if(result_plus){
              matrix_result.set_element(Row, Col, cuda_min<T>(Cvalue, matrix_result.get_element(Row, Col)));
            }else{
              matrix_result.set_element(Row, Col, Cvalue);
            }
        }
      }

      /**
       * \brief R-Kleene main loop.
       * \param distance_matrix The matrix to be saved.
       * \param infinite The max distance that can be stored.
       */
      template <typename T>
      void r_kleene_loop(vnegpu::util::matrix<T>* distance_matrix, T infinite)
      {
          if(distance_matrix->num_linhas<CUDA_BLOCK_SIZE_SHARED)
          {
            //here is always a square matrix
            dim3 threads(distance_matrix->num_linhas, distance_matrix->num_linhas);
            dim3 grid(1, 1);
            apsp_small_matrix<<<grid, threads>>>(*distance_matrix);
          }else{
            int new_size_row = distance_matrix->num_linhas/2;
            int new_size_col = distance_matrix->num_colunas/2;
            vnegpu::util::matrix<T>* A = distance_matrix->sub_matrix(0, 0, new_size_row, new_size_col);
            vnegpu::util::matrix<T>* B = distance_matrix->sub_matrix(0, new_size_col, new_size_row, distance_matrix->num_colunas-new_size_col);
            vnegpu::util::matrix<T>* C = distance_matrix->sub_matrix(new_size_row, 0, distance_matrix->num_linhas-new_size_row, new_size_col);
            vnegpu::util::matrix<T>* D = distance_matrix->sub_matrix(new_size_row, new_size_col, distance_matrix->num_linhas-new_size_row, distance_matrix->num_colunas-new_size_col);

            dim3 block_A(CUDA_BLOCK_SIZE_SHARED, CUDA_BLOCK_SIZE_SHARED);
            dim3 grid_A( (A->num_colunas-1) / block_A.x + 1, (A->num_linhas-1) / block_A.y + 1);

            dim3 block_B(CUDA_BLOCK_SIZE_SHARED, CUDA_BLOCK_SIZE_SHARED);
            dim3 grid_B( (B->num_colunas-1) / block_B.x + 1, (B->num_linhas-1) / block_B.y + 1);

            dim3 block_C(CUDA_BLOCK_SIZE_SHARED, CUDA_BLOCK_SIZE_SHARED);
            dim3 grid_C( (C->num_colunas-1) / block_C.x + 1, (C->num_linhas-1) / block_C.y + 1);

            dim3 block_D(CUDA_BLOCK_SIZE_SHARED, CUDA_BLOCK_SIZE_SHARED);
            dim3 grid_D( (D->num_colunas-1) / block_D.x + 1, (D->num_linhas-1) / block_D.y + 1);

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

            matrix_mul_min_kernel<<<grid_B, block_B>>>(*A, *B, *B, false, infinite);
            CUDA_CHECK();
            //cudaDeviceSynchronize();

            matrix_mul_min_kernel<<<grid_C, block_C>>>(*C, *A, *C, false, infinite);
            CUDA_CHECK();

            matrix_mul_min_kernel<<<grid_D, block_D>>>(*C, *B, *D, true, infinite);
            CUDA_CHECK();

            r_kleene_loop(D, infinite);

            matrix_mul_min_kernel<<<grid_B, block_B>>>(*B, *D, *B, false, infinite);
            CUDA_CHECK();

            matrix_mul_min_kernel<<<grid_C, block_C>>>(*D, *C, *C, false, infinite);
            CUDA_CHECK();

            matrix_mul_min_kernel<<<grid_A, block_A>>>(*B, *C, *A, true, infinite);
            CUDA_CHECK();

            cudaDeviceSynchronize();
            CUDA_CHECK();

            delete A;
            delete B;
            delete C;
            delete D;
          }

      }

      /**
       * \brief Kernel that generate the matrix of distances to be used on the R-Kleene Algorithm.
       * \param graph_distance The graph to be used.
       * \param distance_index The index of the adge variable used as distance.
       * \param distance_matrix The matrix to be saved.
       * \param infinite The max distance that can be stored.
       */
      template <typename T, class VariablesType>
      __global__ void construct_distance_matrix(graph<T,VariablesType> graph_distance,
                                                vnegpu::util::matrix<T> distance_matrix,
                                                int distance_index,
                                                T infinite)
      {
        int id = threadIdx.x + blockIdx.x * blockDim.x;

        if(id >= graph_distance.get_num_nodes())
             return;

         for(int i=0;i<graph_distance.get_num_nodes();i++){
           if(i==id){
             distance_matrix.set_element(id,i,0);
           }else{
             distance_matrix.set_element(id,i,infinite);
           }
         }

         for(int i=graph_distance.get_source_offset(id); i<graph_distance.get_source_offset(id+1); i++)
         {
           int destination = graph_distance.get_destination_indice(i);

           T distance = graph_distance.get_variable_edge(distance_index, i);
           distance_matrix.set_element(id, destination, distance);
         }

      }

      /**
       * \brief Calculate the distance from all nodes to all nodes.
       * \param graph_distance The graph to be used.
       * \param distance_index The index of the adge variable used as distance.
       */
      template <typename T, class VariablesType>
      void r_kleene_imp(graph<T,VariablesType> *graph_distance, int distance_index)
      {
          if(distance_index >= graph_distance->get_num_var_edges()){
            throw std::invalid_argument("The Distance Index is invalid.");
          }

          vnegpu::util::matrix<T>* distance_matrix = new vnegpu::util::matrix<T>(graph_distance->get_num_nodes(),graph_distance->get_num_nodes());

          int num = graph_distance->get_num_nodes()/CUDA_BLOCK_SIZE + 1;
          dim3 Block(CUDA_BLOCK_SIZE);
          dim3 Grid(num);

          T infinite = std::numeric_limits<T>::max();

          construct_distance_matrix<<<Grid, Block>>>(*graph_distance, *distance_matrix, distance_index, infinite);

          //distance_matrix->host_debug_print();

          r_kleene_loop<T>(distance_matrix, infinite); CUDA_CHECK();

          graph_distance->set_distance_matrix(distance_matrix);

          //distance_matrix->host_debug_print();
          cudaDeviceSynchronize(); CUDA_CHECK();

      }
    }//end detail
  }//end algorithm
}//end vnegpu



#endif
