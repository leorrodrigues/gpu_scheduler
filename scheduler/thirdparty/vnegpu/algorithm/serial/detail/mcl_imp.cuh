#ifndef _SERIAL_MCL_IMP_CUH
#define _SERIAL_MCL_IMP_CUH

/*! \file
 *  \brief Serial MCL Core/Implementation functions
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

        template <typename T>
        void matrix_mul(vnegpu::util::host_matrix<T>* matrix_a,
                        vnegpu::util::host_matrix<T>* matrix_b,
                        vnegpu::util::host_matrix<T>* matrix_result)
        {
          for (int i = 0; i < matrix_a->num_linhas; i++)
          {
            for (int j = 0; j < matrix_b->num_colunas; j++)
            {
              T final_value = 0;
              for (int k = 0; k < matrix_a->num_colunas; k++)
              {
                T val_a = matrix_a->get_element(i, k);
                T val_b = matrix_b->get_element(k, j);
                final_value += val_a * val_b;
              }
              matrix_result->set_element(i, j, final_value);
            }
          }
        }

        template <typename T, class VariablesType>
        void process_matrix_groups(graph<T,VariablesType>* graph_distance,
                                   vnegpu::util::host_matrix<T>* result_matrix)
        {
            for(int id=0;id<result_matrix->num_colunas;id++){
              for(int i=0;i<result_matrix->num_linhas;i++)
              {
                if(result_matrix->get_element(i, id)>=(T)0.5){
                  graph_distance->set_group_id(id, i);
                }
              }
            }
        }

        template <typename T>
        void sum_colums(vnegpu::util::host_matrix<T>* distance_matrix, T* colum_sum, float r_factor )
        {
            for(int id=0;id<distance_matrix->num_colunas;id++){
              colum_sum[id]=(T)0;
              for(int i=0;i<distance_matrix->num_linhas;i++)
              {
                colum_sum[id]+=cuda_pow<T>(distance_matrix->get_element(i, id),r_factor);
              }
            }
        }

        template <typename T>
        void mcl_inflation(vnegpu::util::host_matrix<T>* in, vnegpu::util::host_matrix<T>* out, T* colum_sum, float r_factor)
        {
            for(int id=0;id<in->num_colunas;id++){
              for(int i=0;i<in->num_linhas;i++)
              {
                out->set_element(i, id, cuda_pow<T>(in->get_element(i, id), r_factor) / colum_sum[id] );
              }
            }

        }

        template <typename T, class VariablesType>
        void construct_stocastic_matrix(graph<T,VariablesType>* graph_distance,
                                        vnegpu::util::host_matrix<T>* distance_matrix,
                                        int distance_index)
        {
          for(int id=0;id<graph_distance->get_num_nodes();id++){
             T soma = 0;
             T maior = 0;
             T atual;

             for(int i=graph_distance->get_source_offset(id); i<graph_distance->get_source_offset(id+1); i++)
             {
               atual = graph_distance->get_variable_edge(distance_index, i);
               soma += atual;
               if(atual>maior){
                 maior=atual;
               }
             }

             //Self Loop Edge
             soma+=maior;

             for(int i=graph_distance->get_source_offset(id); i<graph_distance->get_source_offset(id+1); i++)
             {
               int destination = graph_distance->get_destination_indice(i);

               T distance = graph_distance->get_variable_edge(distance_index, i);

               distance_matrix->set_element(destination, id, distance/soma);
             }

             //Self Loop Edge
             distance_matrix->set_element(id, id, maior/soma);
          }

        }

        template <typename T>
        void check_iteration(vnegpu::util::host_matrix<T>* now,
                             vnegpu::util::host_matrix<T>* before,
                             T max_difference,
                             bool* d_run)
        {
          for(int id=0;id<now->num_colunas;id++){
            for(int i=0;i<now->num_linhas;i++)
            {
              if(now->get_element(i, id)-before->get_element(i, id)>max_difference){
                d_run[0]=true;
              }
            }
          }
        }

        /**
         * \brief Classify the nodes of the graph in groups based on a distance.
         * \param graph_distance The graph to be used.
         */
         template <typename T, class VariablesType>
         void mcl_imp(graph<T,VariablesType> *graph_group, int distance_index, float p_factor, float r_factor, T max_error)
         {
            graph_group->initialize_group();

            vnegpu::util::host_matrix<T>* distance_matrix = new vnegpu::util::host_matrix<T>(graph_group->get_num_nodes(),graph_group->get_num_nodes());

            construct_stocastic_matrix(graph_group, distance_matrix, distance_index);

            vnegpu::util::host_matrix<T>* temp = new vnegpu::util::host_matrix<T>(distance_matrix->num_linhas, distance_matrix->num_colunas);


            T* colum_sum = (T*)malloc(sizeof(T)*graph_group->get_num_nodes());

             int t=500;

             bool run=true;

             T* change;

             //printf("-\n");

              while(t-- && run){

                matrix_mul(distance_matrix, distance_matrix, temp);

                sum_colums(temp, colum_sum, r_factor);

                mcl_inflation(temp, temp, colum_sum, r_factor);

                run=false;

                check_iteration(temp, distance_matrix, max_error, &run);

                change = distance_matrix->data;
                distance_matrix->data = temp->data;
                temp->data = change;

              }

            process_matrix_groups(graph_group, distance_matrix);

            //distance_matrix->print();

         }
      }//end detail
    }//end serial
  }//end algorithm
}//end vnegpu



#endif
