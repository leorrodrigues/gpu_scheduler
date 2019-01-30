
#ifndef _TOPSIS_KERNEL_NOT_INCLUDED_
#define _TOPSIS_KERNEL_NOT_INCLUDED_

#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <chrono>

static __global__
void powKernel(float* matrix, float* aux_matrix, int alt_size, int resources_size){
	//Colocar Shared Memory Para otimizar
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	if( x < alt_size && y < resources_size) {
		aux_matrix[x*resources_size+y]= matrix[x*resources_size+y] * matrix[x*resources_size+y];
	}
}

static __global__
void sumLinesSqrtKernel(float* aux_matrix, float* vec, int alt_size, int resources_size){
	int gid_x = blockIdx.x*blockDim.x+threadIdx.x;
	if( gid_x < resources_size) {
		int i;
		for( i=0; i<alt_size; i++) {
			vec[gid_x] += aux_matrix[gid_x*alt_size+i];
		}
		vec[gid_x] = sqrtf(vec[gid_x]);
	}
}

static __global__
void normalizeKernel(float* matrix, float* weights, float* vec, int alt_size, int resources_size){
	int gid_x = blockIdx.x*blockDim.x+threadIdx.x;
	int gid_y = blockIdx.y*blockDim.y+threadIdx.y;
	if( gid_x < resources_size && gid_y < alt_size) {
		matrix[gid_x*alt_size+gid_y] /= vec[gid_x];
		matrix[gid_x*alt_size+gid_y] *= weights[gid_x];
	}
}

template <unsigned int blockSize>
static __device__ void warpMaxReduce(volatile float *sdata, unsigned int tid){
	if(blockSize >=64) sdata[tid] = sdata[tid] > sdata[tid+32] ? sdata[tid] : sdata[tid+32];
	if(blockSize >=32) sdata[tid] = sdata[tid] > sdata[tid+16] ? sdata[tid] : sdata[tid+16];
	if(blockSize >=16) sdata[tid] = sdata[tid] > sdata[tid+ 8] ? sdata[tid] : sdata[tid+ 8];
	if(blockSize >= 8) sdata[tid] = sdata[tid] > sdata[tid+ 4] ? sdata[tid] : sdata[tid+ 4];
	if(blockSize >= 4) sdata[tid] = sdata[tid] > sdata[tid+ 2] ? sdata[tid] : sdata[tid+ 2];
	if(blockSize >= 2) sdata[tid] = sdata[tid] > sdata[tid+ 1] ? sdata[tid] : sdata[tid+ 1];
}

template <unsigned int blockSize=512> __global__
static void maxKernelReduction(float *data, float *max, unsigned int size){
	extern __shared__ float sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2)+tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid]=-1;

	if(i>=size) return;

	while(i<size) {
		sdata[tid] = data[tid] > data[i] ? data[tid] : data[i];
		if(i+blockSize<size)
			sdata[tid] = sdata[tid] > data[i+blockSize] ? sdata[tid] : data[i+blockSize];
		i+=gridSize;
	}
	__syncthreads();

	if(blockSize >=512) {
		if(tid < 256 ) {
			sdata[tid] = sdata[tid] > sdata[tid+256] ? sdata[tid] : sdata[tid+256];
		}
		__syncthreads();
	}
	if(blockSize >=256) {
		if(tid < 128 ) {
			sdata[tid] = sdata[tid] > sdata[tid+128] ? sdata[tid] : sdata[tid+128];
		}
		__syncthreads();
	}
	if(blockSize >=128) {
		if(tid < 64 ) {
			sdata[tid] = sdata[tid] > sdata[tid+64] ? sdata[tid] : sdata[tid+64];
		}
		__syncthreads();
	}
	if(tid<32) warpMaxReduce<32>(sdata, tid);
	if(tid==0) max[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
static __device__ void warpMinReduce(volatile float *sdata, unsigned int tid){
	if(blockSize >=64) sdata[tid] = sdata[tid] > sdata[tid+32] ? sdata[tid+32] : sdata[tid];
	if(blockSize >=32) sdata[tid] = sdata[tid] > sdata[tid+16] ? sdata[tid+16] : sdata[tid];
	if(blockSize >=16) sdata[tid] = sdata[tid] > sdata[tid+ 8] ? sdata[tid+ 8] : sdata[tid];
	if(blockSize >= 8) sdata[tid] = sdata[tid] > sdata[tid+ 4] ? sdata[tid+ 4] : sdata[tid];
	if(blockSize >= 4) sdata[tid] = sdata[tid] > sdata[tid+ 2] ? sdata[tid+ 2] : sdata[tid];
	if(blockSize >= 2) sdata[tid] = sdata[tid] > sdata[tid+ 1] ? sdata[tid+ 1] : sdata[tid];
}

template <unsigned int blockSize=512> __global__
static void minKernelReduction(float *data, float *min, unsigned int size){
	extern __shared__ float sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2)+tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid]=FLT_MAX;

	if(i>=size) return;

	while(i<size) {
		sdata[tid] = data[tid] > data[i] ? data[i] : data[tid];
		if(i+blockSize<size)
			sdata[tid] = sdata[tid] > data[i+blockSize] ? data[i+blockSize] : sdata[tid];
		i+=gridSize;
	}
	__syncthreads();

	if(blockSize >=512) {
		if(tid < 256 ) {
			sdata[tid] = sdata[tid] > sdata[tid+256] ? sdata[tid+256] : sdata[tid];
		}
		__syncthreads();
	}
	if(blockSize >=256) {
		if(tid < 128 ) {
			sdata[tid] = sdata[tid] > sdata[tid+128] ? sdata[tid+128] : sdata[tid];
		}
		__syncthreads();
	}
	if(blockSize >=128) {
		if(tid < 64 ) {
			sdata[tid] = sdata[tid] > sdata[tid+64] ? sdata[tid+64] : sdata[tid];
		}
		__syncthreads();
	}
	if(tid<32) warpMinReduce<32>(sdata, tid);
	if(tid==0) {
		min[blockIdx.x] = sdata[0];
	}
}

static void prepareMinMaxKernel(float* d_matrix, float* d_min, float* d_max, unsigned int* types, unsigned int alt_size, unsigned int resources_size){
	int devID;
	cudaDeviceProp props;
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&props, devID);
	int block_size = (props.major <2) ? 16 : 32;
	unsigned int blocks_amount = ceil(alt_size/(float)(block_size*2));

	int i;
	size_t smemSize = block_size*sizeof(float);

	//Host temp array
	float* h_blocks_temp;

	//Malloc host array
	h_blocks_temp = (float*) malloc (sizeof(float)*blocks_amount);

	//Cuda arrays
	float *d_temp_vec, *d_block_max_vec, *d_block_min_vec;

	//Cuda malloc arrays
	cudaMalloc((void**)&d_temp_vec, sizeof(float)*alt_size);
	cudaMalloc((void**)&d_block_max_vec, sizeof(float)*blocks_amount);
	cudaMalloc((void**)&d_block_min_vec, sizeof(float)*blocks_amount);

	//Set the threads per block and blocks per grid sizes
	dim3 dimBlock(block_size);
	dim3 dimGrid(blocks_amount);

	for(i=0; i< resources_size; i++) {
		//get max value
		cudaMemcpy(d_temp_vec, &d_matrix[i * alt_size], sizeof(float) * alt_size, cudaMemcpyDeviceToDevice);

		//The template parameter send has +1 to avoid bank conflits in shmem
		switch(block_size) {
		case 512:
			maxKernelReduction<513><<<dimGrid, dimBlock, smemSize>>>(d_temp_vec, d_block_max_vec, alt_size);
			minKernelReduction<513><<<dimGrid, dimBlock, smemSize>>>(d_temp_vec, d_block_min_vec, alt_size);
			break;
		case 256:
			maxKernelReduction<257><<<dimGrid, dimBlock, smemSize>>>(d_temp_vec, d_block_max_vec, alt_size);
			minKernelReduction<257><<<dimGrid, dimBlock, smemSize>>>(d_temp_vec, d_block_min_vec, alt_size);
			break;
		case 128:
			maxKernelReduction<129><<<dimGrid, dimBlock, smemSize>>>(d_temp_vec, d_block_max_vec, alt_size);
			minKernelReduction<129><<<dimGrid, dimBlock, smemSize>>>(d_temp_vec, d_block_min_vec, alt_size);
			break;
		case 64:
			maxKernelReduction<65><<<dimGrid, dimBlock, smemSize>>>(d_temp_vec, d_block_max_vec, alt_size);
			minKernelReduction<65><<<dimGrid, dimBlock, smemSize>>>(d_temp_vec, d_block_min_vec, alt_size);
			break;
		case 32:
			maxKernelReduction<33><<<dimGrid, dimBlock, smemSize>>>(d_temp_vec, d_block_max_vec, alt_size);
			minKernelReduction<33><<<dimGrid, dimBlock, smemSize>>>(d_temp_vec, d_block_min_vec, alt_size);
			break;
		case 16:
			maxKernelReduction<17><<<dimGrid, dimBlock, smemSize>>>(d_temp_vec, d_block_max_vec, alt_size);
			minKernelReduction<17><<<dimGrid, dimBlock, smemSize>>>(d_temp_vec, d_block_min_vec, alt_size);
			break;
		case 8:
			maxKernelReduction<9><<<dimGrid, dimBlock, smemSize>>>(d_temp_vec, d_block_max_vec, alt_size);
			minKernelReduction<9><<<dimGrid, dimBlock, smemSize>>>(d_temp_vec, d_block_min_vec, alt_size);
			break;
		case 4:
			maxKernelReduction<5><<<dimGrid, dimBlock, smemSize>>>(d_temp_vec, d_block_max_vec, alt_size);
			minKernelReduction<5><<<dimGrid, dimBlock, smemSize>>>(d_temp_vec, d_block_min_vec, alt_size);
			break;
		case 2:
			maxKernelReduction<3><<<dimGrid, dimBlock, smemSize>>>(d_temp_vec, d_block_max_vec, alt_size);
			minKernelReduction<3><<<dimGrid, dimBlock, smemSize>>>(d_temp_vec, d_block_min_vec, alt_size);
			break;
		case 1:
			maxKernelReduction<2><<<dimGrid, dimBlock, smemSize>>>(d_temp_vec, d_block_max_vec, alt_size);
			minKernelReduction<2><<<dimGrid, dimBlock, smemSize>>>(d_temp_vec, d_block_min_vec, alt_size);
			break;
		}

		if(types[i]) {
			cudaMemcpy(h_blocks_temp, d_block_max_vec, sizeof(float)*blocks_amount, cudaMemcpyDeviceToHost);

			cudaMemcpy(
				&d_max[i],
				std::max_element(
					h_blocks_temp,
					h_blocks_temp+blocks_amount
					),
				sizeof(float),
				cudaMemcpyHostToDevice);

			cudaMemcpy(h_blocks_temp, d_block_min_vec, sizeof(float)*blocks_amount, cudaMemcpyDeviceToHost);

			cudaMemcpy(
				&d_min[i],
				std::min_element(
					h_blocks_temp,
					h_blocks_temp+blocks_amount
					),
				sizeof(float),
				cudaMemcpyHostToDevice);
		}else{
			cudaMemcpy(h_blocks_temp, d_block_max_vec, sizeof(float)*blocks_amount, cudaMemcpyDeviceToHost);

			cudaMemcpy(
				&d_min[i],
				std::max_element(
					h_blocks_temp,
					h_blocks_temp+blocks_amount
					),
				sizeof(float),
				cudaMemcpyHostToDevice);

			cudaMemcpy(h_blocks_temp, d_block_min_vec, sizeof(float)*blocks_amount, cudaMemcpyDeviceToHost);

			cudaMemcpy(
				&d_max[i],
				std::min_element(
					h_blocks_temp,
					h_blocks_temp+blocks_amount
					),
				sizeof(float),
				cudaMemcpyHostToDevice);
		}
	}

	free(h_blocks_temp);
	cudaFree(d_temp_vec);
	cudaFree(d_block_max_vec);
	cudaFree(d_block_min_vec);
}

static void testMaxKernel(){
	int devID;
	cudaDeviceProp props;
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&props, devID);
	int block_size = (props.major <2) ? 16 : 32;

	size_t smemSize = block_size*sizeof(float);

	const int sizes[]={10,100,1000,10000,100000,1000000,10000000, 784000000};

	printf("Randon Test\n");
	for(int k=0; k<8; k++) {
		srand (time(NULL));

		cudaEvent_t start, stop;
		float elapsedTime;
		float resultado;

		unsigned int blocks_amount = ceil(sizes[k]/(float)(block_size*2));
		float *test = (float*) malloc (sizeof(float)*(sizes[k]));
		float *blocks = (float*) malloc (sizeof(float)*blocks_amount);
		float *d_vec, *d_block_vec, d_result;

		cudaMalloc((void**)&d_vec, sizeof(float)*(sizes[k]));
		cudaMalloc((void**)&d_block_vec, sizeof(float)*blocks_amount);

		dim3 dimBlock(block_size);
		dim3 dimGrid(blocks_amount);

		for(int i=0; i<sizes[k]; i++) {
			test[i]=rand() % 10000000 + 1;
		}

		std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
		resultado= *std::max_element(test, test+sizes[k]);
		std::chrono::high_resolution_clock::time_point stop_time = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count();

		cudaMemcpy(d_vec, test, sizeof(float) * sizes[k], cudaMemcpyHostToDevice);

		cudaEventCreate(&start);
		cudaEventRecord(start,0);

		maxKernelReduction<<<dimGrid, dimBlock, smemSize>>>(d_vec, d_block_vec, sizes[k]);

		cudaEventCreate(&stop);
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&elapsedTime, start,stop);
		printf("STD Elapsed time  (%10d) : ",sizes[k]);
		std::cout<< duration/1000.0 << " ms\n";
		printf("CUDA Elapsed time (%10d) : %f ms\n",sizes[k], elapsedTime);

		//cudaMemcpy(&d_result, d_block_vec,sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(blocks, d_block_vec, sizeof(float)*blocks_amount, cudaMemcpyDeviceToHost);
		d_result = *std::max_element(blocks, blocks+blocks_amount);

		free(test);
		free(blocks);
		cudaFree(d_vec);
		cudaFree(d_block_vec);

		if(d_result!=resultado) {
			printf("RESULTS NOT MATCH! MAX FUNCTION NOT WORKS\n");
			printf("%d $ %f\n",resultado, d_result);
		}
	}
}

static void testMinKernel(){
	int devID;
	cudaDeviceProp props;
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&props, devID);
	int block_size = (props.major <2) ? 16 : 32;

	size_t smemSize = block_size*sizeof(float);

	const int sizes[]={10,100,1000,10000,100000,1000000,10000000, 784000000};

	printf("Randon Test\n");
	for(int k=0; k<8; k++) {
		srand (time(NULL));

		cudaEvent_t start, stop;
		float elapsedTime;
		float resultado;

		unsigned int blocks_amount = ceil(sizes[k]/(float)(block_size*2));
		float *test = (float*) malloc (sizeof(float)*(sizes[k]));
		float *blocks = (float*) malloc (sizeof(float)*blocks_amount);
		float *d_vec, *d_block_vec, d_result;

		cudaMalloc((void**)&d_vec, sizeof(float)*(sizes[k]));
		cudaMalloc((void**)&d_block_vec, sizeof(float)*blocks_amount);

		dim3 dimBlock(block_size);
		dim3 dimGrid(blocks_amount);

		for(int i=0; i<sizes[k]; i++) {
			test[i]=rand() % 10000000 + 1;
		}
		printf("\n");

		std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
		resultado= *std::min_element(test, test+sizes[k]);
		std::chrono::high_resolution_clock::time_point stop_time = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count();

		cudaMemcpy(d_vec, test, sizeof(float) * sizes[k], cudaMemcpyHostToDevice);

		cudaEventCreate(&start);
		cudaEventRecord(start,0);

		minKernelReduction<<<dimGrid, dimBlock, smemSize>>>(d_vec, d_block_vec, sizes[k]);

		cudaEventCreate(&stop);
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&elapsedTime, start,stop);
		printf("STD Elapsed time  (%10d) : ",sizes[k]);
		std::cout<< duration/1000.0 << " ms\n";
		printf("CUDA Elapsed time (%10d) : %f ms\n",sizes[k], elapsedTime);

		//cudaMemcpy(&d_result, d_block_vec,sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(blocks, d_block_vec, sizeof(float)*blocks_amount, cudaMemcpyDeviceToHost);
		d_result = *std::min_element(blocks, blocks+blocks_amount);

		free(test);
		free(blocks);
		cudaFree(d_vec);
		cudaFree(d_block_vec);

		if(d_result!=resultado) {
			printf("RESULTS NOT MATCH! MAX FUNCTION NOT WORKS\n");
			printf("%d $ %f\n",resultado, d_result);
			for(int i=0; i< blocks_amount; i++) {
				printf("%f - ", blocks[i]);
			}
			printf("\n");
			cudaDeviceReset();
			exit(0);
		}
	}
}

static __global__
void subtractByMinMaxKernel(float* matrix, float* max_matrix, float* min_matrix, float* max_vec, float* min_vec, int alt_size, int resources_size){
	int gid_x = blockIdx.x*blockDim.x+threadIdx.x;
	int gid_y = blockIdx.y*blockDim.y+threadIdx.y;
	if( gid_x < alt_size && gid_y < resources_size) {
		max_matrix[gid_y*alt_size+gid_x] = matrix[gid_y*alt_size+gid_x] - max_vec[gid_y];

		max_matrix[gid_y*alt_size+gid_x] *= max_matrix[gid_y*alt_size+gid_x];

		min_matrix[gid_y*alt_size+gid_x] = matrix[gid_y*alt_size+gid_x] - min_vec[gid_y];

		min_matrix[gid_y*alt_size+gid_x] *= min_matrix[gid_y*alt_size+gid_x];
	}
}

static __global__
void ldKernel(float* max_matrix, float* min_matrix, float* max_vec, float* min_vec, int alt_size, int resources_size){
	int gid_x = blockIdx.x*blockDim.x+threadIdx.x;
	if( gid_x < alt_size) {
		int i;
		for( i=0; i<resources_size; i++) {
			max_vec[gid_x] += max_matrix[i*alt_size+gid_x];
			min_vec[gid_x] += min_matrix[i*alt_size+gid_x];
		}
		max_vec[gid_x] = sqrtf(max_vec[gid_x]);
		min_vec[gid_x] = sqrtf(min_vec[gid_x]);
	}
}

static __global__
void performanceScoreKernel(float *d_smax, float *d_smin, float *d_result, int size){
	int gid_x = blockIdx.x*blockDim.x+threadIdx.x;
	if( gid_x < size) {
		d_result[gid_x] = d_smin[gid_x]/(d_smax[gid_x]+d_smin[gid_x]);
	}
}
#endif
