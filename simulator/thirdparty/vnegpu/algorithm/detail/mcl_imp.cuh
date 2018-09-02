#ifndef _MCL_IMP_CUH
#define _MCL_IMP_CUH

/*! \file
 *  \brief MCL Core/Implementation functions
 */

#include <limits>

#include <vnegpu/graph.cuh>

#include <vnegpu/util/util.cuh>
#include <vnegpu/util/matrix.cuh>

#include <cublas_v2.h>

namespace vnegpu
{
namespace algorithm
{
namespace detail
{

template <typename T, class VariablesType>
__global__ void process_matrix_groups(graph<T,VariablesType> graph_distance, vnegpu::util::matrix<T> result_matrix)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if(id >= result_matrix.num_colunas)
		return;

	bool nota=true;
	float maior=0;
	for(int i=0; i<result_matrix.num_linhas; i++)
	{
		if(result_matrix.get_element(i, id)>=maior) {
			maior = result_matrix.get_element(i, id);
			graph_distance.set_group_id(id, i);
			nota=false;
		}
		//printf("RESULT %d\n ",result_matrix.get_element(i,id));
	}
	if(nota) {
		printf("ERROZAO:%d\n",id);
	}

}

template <typename T>
__global__ void sum_colums(vnegpu::util::matrix<T> distance_matrix, T* colum_sum, float r_factor )
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if(id >= distance_matrix.num_colunas)
		return;

	colum_sum[id]=(T)0;
	for(int i=0; i<distance_matrix.num_linhas; i++)
	{
		colum_sum[id]+=cuda_pow<T>(distance_matrix.get_element(i, id),r_factor);
	}

	//printf("[%d]=%f\n",id,colum_sum[id]);
}

template <typename T>
__global__ void mcl_inflation(vnegpu::util::matrix<T> in, vnegpu::util::matrix<T> out, T* colum_sum, float r_factor)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if(id >= in.num_colunas)
		return;


	for(int i=0; i<in.num_linhas; i++)
	{
		out.set_element(i, id, cuda_pow<T>(in.get_element(i, id), r_factor) / colum_sum[id] );
	}

}

template <typename T, class VariablesType>
__global__ void construct_stocastic_matrix(graph<T,VariablesType> graph_distance, vnegpu::util::matrix<T> distance_matrix, int distance_index)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if(id >= graph_distance.get_num_nodes())
		return;

	T soma = 0;
	T maior = 0;
	T atual;

	for(int i=graph_distance.get_source_offset(id); i<graph_distance.get_source_offset(id+1); i++)
	{
		atual = graph_distance.get_variable_edge(distance_index, i);
		soma += atual;
		if(atual>maior) {
			maior=atual;
		}
	}

	//Self Loop Edge
	soma+=maior;

	for(int i=graph_distance.get_source_offset(id); i<graph_distance.get_source_offset(id+1); i++)
	{
		int destination = graph_distance.get_destination_indice(i);

		T distance = graph_distance.get_variable_edge(distance_index, i);

		distance_matrix.set_element(destination, id, distance/soma);
	}

	//Self Loop Edge
	distance_matrix.set_element(id, id, maior/soma);

}

template <typename T>
__global__ void check_iteration(vnegpu::util::matrix<T> now, vnegpu::util::matrix<T> before, T max_difference, bool* d_run)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if(id >= now.num_colunas)
		return;


	for(int i=0; i<now.num_linhas; i++)
	{
		if(now.get_element(i, id)-before.get_element(i, id)>max_difference) {
			d_run[0]=true;
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

	vnegpu::util::matrix<T>* distance_matrix = new vnegpu::util::matrix<T>(graph_group->get_num_nodes(),graph_group->get_num_nodes());

	int num = graph_group->get_num_nodes()/CUDA_BLOCK_SIZE + 1;
	dim3 Block(CUDA_BLOCK_SIZE);
	dim3 Grid(num);

	cudaMemset(distance_matrix->data, 0, distance_matrix->pitch*graph_group->get_num_nodes());

	construct_stocastic_matrix<<<Grid, Block>>>(*graph_group, *distance_matrix, distance_index);

	vnegpu::util::matrix<T>* temp = new vnegpu::util::matrix<T>(distance_matrix->num_linhas, distance_matrix->num_colunas);


	dim3 dimBlock(CUDA_BLOCK_SIZE_SHARED, CUDA_BLOCK_SIZE_SHARED);
	dim3 dimGrid((distance_matrix->num_colunas-1) / dimBlock.x + 1, (distance_matrix->num_linhas-1) / dimBlock.y + 1);


	dim3 sum_block(CUDA_BLOCK_SIZE);
	dim3 sum_grid((temp->num_colunas-1) / sum_block.x + 1);


	T* colum_sum;
	cudaMalloc(&colum_sum, sizeof(T)*graph_group->get_num_nodes());
	CUDA_CHECK();


	cublasHandle_t handle;
	cublasCreate(&handle);

	const float a = 1.0f;
	const float b = 0;

	int t=500;

	bool run=true;
	bool* d_run;
	cudaMalloc(&d_run, sizeof(bool));

	T* change;
	while(t-- && run) {
		//printf("Ite:%d\n",t);
		//distance_matrix->host_debug_print();
		//vnegpu::util::matrix_mul_kernel<<<dimGrid, dimBlock>>>(*distance_matrix, *distance_matrix, *temp);
		//cudaDeviceSynchronize();
		//CUDA_CHECK();
		int x=cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, distance_matrix->num_colunas,
		                  distance_matrix->num_colunas, distance_matrix->num_colunas, &a,
		                  distance_matrix->data, distance_matrix->pitch/sizeof(float),
		                  distance_matrix->data, distance_matrix->pitch/sizeof(float),
		                  &b, temp->data, temp->pitch/sizeof(float));
		//CUDA_CHECK();
		//printf("R:%d\n",x);
		//printf("-\n");
		//temp->host_debug_print();
		//printf("- R FCTOR:%f\n",r_factor);

		sum_colums<<<sum_grid, sum_block>>>(*temp, colum_sum, r_factor);

		mcl_inflation<<<sum_grid, sum_block>>>(*temp, *temp, colum_sum, r_factor);

		cudaMemset(d_run, 0, sizeof(bool));

		//distance_matrix->host_debug_print();
		//cudaDeviceSynchronize();
		//printf("--\n");
		//temp->host_debug_print();
		//cudaDeviceSynchronize();
		check_iteration<<<sum_grid, sum_block>>>(*temp, *distance_matrix, max_error, d_run);

		cudaMemcpy(&run,d_run,sizeof(bool),cudaMemcpyDeviceToHost);
		change = distance_matrix->data;
		distance_matrix->data = temp->data;
		temp->data = change;

		cudaDeviceSynchronize();
	}



	//distance_matrix->host_debug_print();
	//cudaDeviceSynchronize();

	process_matrix_groups<<<sum_grid, sum_block>>>(*graph_group, *distance_matrix);
	cudaDeviceSynchronize();
	cudaFree(colum_sum);
	cudaFree(d_run);
	cublasDestroy(handle);
	temp->free();
	distance_matrix->free();
	delete distance_matrix;
	delete temp;

}
}    //end detail
}  //end algorithm
}//end vnegpu



#endif
