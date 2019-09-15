#ifndef _MATRIX_CUH
#define _MATRIX_CUH

#include <stdio.h>
#include "../config.cuh"

/*! \file
 *  \brief Personalised Matrix data struct.
 */

namespace vnegpu {
namespace util {

template <typename T>
struct matrix;

__global__ static void kernel_host_debug_fill_inc(matrix<float> m);

__global__ static void kernel_host_debug_print(matrix<float> m);

__global__ static void kernel_host_debug_fill_one(matrix<float> m);

template <typename T>
__global__ void matrix_mul_kernel(matrix<T> A, matrix<T> B, matrix<T> C);

/**
 * \brief Matrix data structure base on Nvidia implementation on cuda guide.
 */
template <typename T>
struct matrix
{
	int num_colunas;
	int num_linhas;
	size_t pitch;
	T* data;


	__host__ matrix(const unsigned int num_linhas, const unsigned int num_colunas)
	{
		this->num_colunas = num_colunas;
		this->num_linhas = num_linhas;
		cudaMallocPitch(&data, &pitch, sizeof(T)*num_colunas, num_linhas);
		cudaMemset(data, 0, pitch*num_linhas);
		CUDA_CHECK();
	}

	__host__ matrix(const unsigned int num_linhas, const unsigned int num_colunas, T* data, size_t pitch)
	{
		this->num_colunas = num_colunas;
		this->num_linhas = num_linhas;
		this->data = data;
		this->pitch = pitch;
	}

	//TODO:Should free here?
	__host__ ~matrix()
	{

	}

	__host__ void free()
	{
		cudaFree(data);
	}

	__host__ matrix<T>* mul(matrix<T>* b)
	{
		matrix<T>* c = new matrix(this->num_linhas, b->num_colunas);


		dim3 dimBlock(CUDA_BLOCK_SIZE_SHARED, CUDA_BLOCK_SIZE_SHARED);
		dim3 dimGrid( (c->num_colunas-1) / dimBlock.x + 1, (c->num_linhas-1) / dimBlock.y + 1);
		matrix_mul_kernel<<<dimGrid, dimBlock>>>(*this, *b, *c);
		return c;
	}

	// Get a matrix element
	__device__ inline T get_element(const unsigned int row, const unsigned int col)
	{
		T* pElement = (T*)((char*)data + row * pitch) + col;
		return *pElement;
	}

	// Get a matrix element poiter
	__device__ inline T* get_element_poiter(const unsigned int row, const unsigned int col)
	{
		T* pElement = (T*)((char*)data + row * pitch) + col;
		return pElement;
	}

	// Set a matrix element
	__device__ inline void set_element(const unsigned int row, const unsigned int col, T value)
	{
		T* pElement = (T*)((char*)data + row * pitch) + col;
		*pElement = value;
	}


	__host__ void host_debug_fill_print()
	{
		//stupid invocations, meaninfull debug.
		kernel_host_debug_fill_inc<<<1,1>>>(*this);
		kernel_host_debug_print<<<1,1>>>(*this);
		cudaDeviceSynchronize();
	}

	__host__ void host_debug_print()
	{
		//stupid invocations, meaninfull debug.
		kernel_host_debug_print<<<1,1>>>(*this);
		cudaDeviceSynchronize();
	}


	__host__ void host_debug_fill_one()
	{
		kernel_host_debug_fill_one<<<1,1>>>(*this);
	}

	__host__ void host_debug_fill_inc()
	{
		kernel_host_debug_fill_inc<<<1,1>>>(*this);
	}

	__host__ matrix<T>* sub_matrix(int row, int col, int size_rows, int size_cols)
	{
		T* p  = (T*)((char*)data + row * pitch) + col;
		return new matrix<T>(size_rows, size_cols, p, pitch);
	}
};

__global__ void kernel_host_debug_fill_inc(matrix<float> m)
{
	float v=0;
	for(int i=0; i<m.num_linhas; i++) {
		for(int j=0; j<m.num_colunas; j++) {
			v++;
			m.set_element(i,j,v);
		}
	}
}

__global__ void kernel_host_debug_fill_one(matrix<float> m)
{
	float v=0;
	for(int i=0; i<m.num_linhas; i++) {
		for(int j=0; j<m.num_colunas; j++) {
			v++;
			m.set_element(i,j,1.0);
		}
	}
}

__global__ void kernel_host_debug_print(matrix<float> m)
{
	for(int i=0; i<m.num_linhas; i++) {
		for(int j=0; j<m.num_colunas; j++) {
			printf("%.1f ",m.get_element(i,j));
		}
		printf("\n");
	}
}


template <typename T>
__global__ void matrix_mul_kernel(matrix<T> A, matrix<T> B, matrix<T> C)
{

	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	T Cvalue = 0;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int Row = blockRow * CUDA_BLOCK_SIZE_SHARED + ty;
	int Col = blockCol * CUDA_BLOCK_SIZE_SHARED + tx;

	__shared__ T As[CUDA_BLOCK_SIZE_SHARED][CUDA_BLOCK_SIZE_SHARED];
	__shared__ T Bs[CUDA_BLOCK_SIZE_SHARED][CUDA_BLOCK_SIZE_SHARED];


	for (int m = 0; m < ( (A.num_colunas-1) / CUDA_BLOCK_SIZE_SHARED + 1); ++m) {

		if (Row < A.num_linhas && m*CUDA_BLOCK_SIZE_SHARED+tx < A.num_colunas) {
			As[ty][tx] = A.get_element(Row, +m*CUDA_BLOCK_SIZE_SHARED+tx);
		}else{
			As[ty][tx] = (T)0;
		}

		if (Col < B.num_colunas && m*CUDA_BLOCK_SIZE_SHARED+ty < B.num_linhas) {
			Bs[ty][tx] = B.get_element(m*CUDA_BLOCK_SIZE_SHARED+ty,Col);
		}else{
			Bs[ty][tx] = (T)0;
		}

		__syncthreads();

	   #pragma unroll
		for (int e = 0; e < CUDA_BLOCK_SIZE_SHARED; ++e) {
			Cvalue += As[ty][e] * Bs[e][tx];
		}

		__syncthreads();
	}

	if (Row < C.num_linhas && Col < C.num_colunas) {
		C.set_element(Row, Col, Cvalue);
	}
}

}  //util end
}//vnegpu end

#endif
