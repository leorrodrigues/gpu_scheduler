
#ifndef _AHPG_KERNEL_NOT_INCLUDED_
#define _AHPG_KERNEL_NOT_INCLUDED_

static __global__
void acquisitonKernel(float* data, float* min_max, float* result, int sheets_size, int alternatives_size){
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	int z = blockIdx.z*blockDim.z+threadIdx.z;
	// printf("Inside the kernel!!\n");
	int i, j, index;
	float temp=0;
	if( x < sheets_size && y < alternatives_size && z < alternatives_size) { //the thread can do the work
		i = y * sheets_size + x;
		j = z * sheets_size + x;
		index = x*sheets_size*alternatives_size+y*alternatives_size+z;
		if(data[i]==data[j]) {
			temp = 1;
		}else{
			if(min_max[ x ]!=-1) {
				temp = ( data[i] - data[j] ) / min_max[ x ];
			}else{
				data[i]>data[j] ? temp = 9.0 : temp = 1.0/ 9.0;
			}
			if(temp == 0 ) {
				temp = 1;
			} else if (temp <0) {
				temp = (-1) / temp;
			}
		}
		result[ index ] = temp;
		printf("Thread Row %d Col %d, inserted in result[%d] = %2f and temp = %2f\n", x, y, index, result[index], temp);
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
		result[j*size+i] = data[j*size+i] / sum[i];
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
