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
void calculateSUM_Row(float* data, float* sum, int size){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<size) {
		float s = 0;
		int j;
		for(j=0; j<size; j++) {
			s+= data[i*size+j];
		}
		sum[i] = s;
	}
}

static __global__
void calculateSUM_Line(float* data, float* sum, int size){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<size) {
		float s = 0;
		int j;
		for(j=0; j<size; j++) {
			s+= data[j*size+i];
		}
		sum[i] = s;
	}
}

static __global__
void calculateNMatrix(float* data, float* sum, float* result, int size){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int j = blockIdx.y*blockDim.y+threadIdx.y;
	// printf("Inside the kernel!!\n");
	if( i < size && j < size) {
		if(sum[i] == 0) result[j*size+i] = 0;
		else result[j*size+i] = data[j*size+i] / sum[i];
	}
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
