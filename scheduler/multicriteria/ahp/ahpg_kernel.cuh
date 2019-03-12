#ifndef _AHPG_KERNEL_NOT_INCLUDED_
#define _AHPG_KERNEL_NOT_INCLUDED_

static __global__
void normalize19Kernel(float* matrix, float *values, size_t alt_size, size_t criteria_size){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int j = blockIdx.y*blockDim.y+threadIdx.y;
	if( i < alt_size && j < criteria_size) {
		// if the min max difference is 0, all the elements in the matrix are equals and their values are set to 1, otherwise the division has to be made.
		matrix[j*alt_size+i] = (values[j]!=0) ?  (matrix[j*alt_size+i]/values[j]) : 0;
	}
}

static __global__
void pairwiseComparsionKernel(float *matrix, float *result, size_t alt_size, size_t criteria_size){
	// extern __shared__ float sdata[];

	// We need to get the values from I, J and K
	// The I and J represents the alternatives while K represents the Criterias
	const unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
	const unsigned int j = blockIdx.y*blockDim.y+threadIdx.y;
	const unsigned int k = blockIdx.z*blockDim.z+threadIdx.z;
	// Check the bounds
	if( i >= alt_size || j >= alt_size || k >= criteria_size) return;

	size_t index_out = k*alt_size*alt_size+j*alt_size+i;

	// sdata[threadIdx.x+blockDim.x*threadIdx.y] = matrix[i*criteria_size+k];
	// __syncthreads();

	// result[index_out] = sdata[threadIdx.x+blockDim.x*threadIdx.z]-sdata[threadIdx.y+blockDim.x*threadIdx.z];
	result[index_out] = matrix[k*alt_size+j]-matrix[k*alt_size+i];
	if(result[index_out]<0) {
		result[index_out] = (-1.0)/result[index_out];
	} else if(result[index_out]==0) {
		result[index_out]=1;
	}
}

static __global__
void sumLinesKernel(float* matrix, float* sum, size_t alt_size, size_t criteria_size){
	const unsigned int x = blockIdx.x*blockDim.x+threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y+threadIdx.y;

	if(x>=alt_size || y>=criteria_size) return;

	sum[y*alt_size+x] = 0;
	for( size_t i=0; i<alt_size; i++) {
		sum[y*alt_size+x] += matrix[y*alt_size*alt_size+x*alt_size+i];
	}
}

static __global__
void normalizeMatrixKernel(float* matrix, float* sum, size_t alt_size, size_t criteria_size){
	const unsigned int x = blockIdx.x*blockDim.x+threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y+threadIdx.y;
	const unsigned int z = blockIdx.z*blockDim.z+threadIdx.z;
	// printf("Inside the kernel!!\n");
	if( x >= alt_size || y >= alt_size || z >= criteria_size) return;

	const size_t index = z*alt_size*alt_size+y*alt_size+x;

	if(sum[z*alt_size+y] == 0) matrix[index] = 0;
	else matrix[index] = matrix[index] / sum[z*alt_size+y];
}

static __global__
void calculateCPml(float* data, float* result, int size){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	// printf("Inside the kernel!!\n");
	if( i < size ) {
		float sum=0;
		int j;
		for(j=0; j<size; j++) {
			sum += data[i*size+j];
		}
		result[i] = sum / (float)size;

	}
}

#endif
