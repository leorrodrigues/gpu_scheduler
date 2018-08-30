#ifndef _HOST_MATRIX_CUH
#define _HOST_MATRIX_CUH

#include <stdio.h>
#include <vnegpu/config.cuh>

/*! \file
 *  \brief Personalised Host Matrix data struct.
 */

namespace vnegpu
{
  namespace util
  {
    /**
     * \brief Matrix data structure for Host Memory.
     */
    template <typename T>
    struct host_matrix
    {
        int num_colunas;
        int num_linhas;
        size_t pitch;
        T* data;


        __host__ host_matrix(const unsigned int num_linhas, const unsigned int num_colunas)
        {
          this->num_colunas = num_colunas;
          this->num_linhas = num_linhas;
          this->pitch = num_colunas*sizeof(T);
          data = (T*)malloc(sizeof(T)*num_colunas*num_linhas);
          memset(data, 0, sizeof(T)*num_colunas*num_linhas);
        }

        __host__ host_matrix(const unsigned int num_linhas, const unsigned int num_colunas, T* data, size_t pitch)
        {
          this->num_colunas = num_colunas;
          this->num_linhas = num_linhas;
          this->data = data;
          this->pitch = pitch;
        }

        __host__ host_matrix(vnegpu::util::matrix<T>* device_matrix)
        {
          this->num_colunas = device_matrix->num_colunas;
          this->num_linhas = device_matrix->num_linhas;
          this->pitch = device_matrix->num_colunas*sizeof(T);
          data = (T*)malloc(sizeof(T)*this->num_colunas*this->num_linhas);

          cudaMemcpy2D(data, this->pitch, device_matrix->data, device_matrix->pitch,
                       this->num_colunas*sizeof(T) , this->num_linhas, cudaMemcpyDeviceToHost);

        }


        //TODO:Should free here?
        __host__ ~host_matrix()
        {

        }

        __host__ void free()
        {
          free(data);
        }

        __host__ host_matrix<T>* mul(host_matrix<T>* b)
        {
          host_matrix<T>* c = new host_matrix(this->num_linhas, b->num_colunas);


          return c;
        }

        // Get a matrix element
        __host__ inline T get_element(const unsigned int row, const unsigned int col)
        {
            T* pElement = (T*)((char*)data + row * pitch) + col;
            return *pElement;
        }

        // Set a matrix element
        __host__ inline void set_element(const unsigned int row, const unsigned int col, T value)
        {
            T* pElement = (T*)((char*)data + row * pitch) + col;
            *pElement = value;
        }


        __host__ host_matrix<T>* sub_matrix(int row, int col, int size_rows, int size_cols)
        {
            T* p  = (T*)((char*)data + row * pitch) + col;
            return new host_matrix<T>(size_rows, size_cols, p, pitch);
        }

        __host__ void print(){
          for(int i=0;i<this->num_linhas;i++){
            for(int j=0;j<this->num_colunas;j++){
              printf("%.1f ",this->get_element(i,j));
            }
            printf("\n");
          }
        }

        __host__ bool
        comp(host_matrix<T>* matrix_b) {

          if(this->num_linhas != matrix_b->num_linhas)
              return false;
          if(this->num_colunas != matrix_b->num_colunas)
              return false;

          for(int i=0;i<this->num_linhas;i++){
            for(int j=0;j<this->num_colunas;j++){
              T element_a = this->get_element(i,j);
              T element_b =  matrix_b->get_element(i,j);
              if(element_a != element_b)
                  return false;
            }
          }

          return true;
        }
    };


  }//util end
}//vnegpu end

#endif
