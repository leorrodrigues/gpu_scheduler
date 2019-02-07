#include "topsis.cuh"

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n",
		        cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

TOPSIS::TOPSIS(){
	char cwd[1024];
	char* result;
	result = getcwd(cwd, sizeof(cwd));
	if(result == NULL) {
		printf("TOPSIS Error get directory path\n");
	}
	char* sub_path = (strstr(cwd, "topsis"));
	if(sub_path!=NULL) {
		int position = sub_path - cwd;
		strncpy(this->path, cwd, position);
		strcat(this->path,"/");
		this->path[position] = '\0';
	}else{
		strcpy(this->path, cwd);
		strcat(this->path,"/multicriteria/\0");
	}
	this->hosts_value = NULL;
	this->hosts_size = 0;
	this->hosts_index = NULL;
}

TOPSIS::~TOPSIS(){
	free(this->hosts_value);
	free(this->hosts_index);
	this->hosts_value=NULL;
	this->hosts_index=NULL;
}

void TOPSIS::getWeights(float* weights, unsigned int* types, std::map<std::string, float> resource){
	char weights_schema_path [1024] = "\0";
	char weights_data_path [1024] = "\0";

	strcpy(weights_schema_path, path);
	strcpy(weights_data_path, path);

	strcat(weights_schema_path, "topsis/json/weightsSchema.json");
	strcat(weights_data_path, "topsis/json/weightsDataFrag.json");

	rapidjson::SchemaDocument weightsSchema = JSON::generateSchema(weights_schema_path);
	rapidjson::Document weightsData = JSON::generateDocument(weights_data_path);
	rapidjson::SchemaValidator weightsValidator(weightsSchema);
	if(!weightsData.Accept(weightsValidator))
		JSON::jsonError(&weightsValidator);

	int i=0;

	float value=0;
	bool type=false;
	int index=0;

	for(auto &dataObject: weightsData.GetObject()) {
		for(auto &arrayObject: dataObject.value.GetArray()) {
			for(auto &data: arrayObject.GetObject()) {
				if(data.value.IsFloat()) {
					value = data.value.GetFloat();
				}else if(data.value.IsString()) {
					index = distance(resource.begin(), resource.find(data.value.GetString()));
					// printf("The resource %s has index %d\n", data.value.GetString(), index);
				}else if(data.value.IsBool()) {
					type = data.value.GetBool();
				}
				i++;
			}
			weights[index] = value;
			types[index] = (type == true) ? 1 : 0;
			// printf("Index = %d\n",index);
		}
	}
	// for(int i=0; i<resource.size(); i++) {
	//      printf("%f - ", weights[i]);
	// }
	// printf("\n");
}

void TOPSIS::run(Host** alternatives, int alt_size){
	// printf("Running topsis\n");
	int devID;
	cudaDeviceProp props;
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&props, devID);
	int block_size = (props.major <2) ? 16 : 32;

	this->hosts_index = (unsigned int*) malloc (sizeof(unsigned int)* alt_size);

	std::map<std::string,float> allResources = alternatives[0]->getResource();

	int resources_size = allResources.size();

	size_t matrix_bytes  = sizeof(float)*resources_size*alt_size;
	size_t weights_bytes = sizeof(float)*(resources_size+1);
	size_t result_bytes  = sizeof(float)*alt_size;

	/*Create the host variables*/
	float *matrix = (float*) malloc (matrix_bytes);
	float *weights= (float*) malloc (weights_bytes);
	unsigned int * types = (unsigned int*) malloc (sizeof(unsigned int)*resources_size);
	// printf("Allocating %d spaces for weights\n", resources_size);
	std::map<std::string, float> a =allResources;
	// for(std::map<std::string, float>::iterator it=a.begin(); it!=a.end(); it++) {
	//      printf("%s %f\n",it->first.c_str(),it->second);
	// }

	/*Create the pinned variables*/
	float *pinned_matrix, *pinned_weights;

	/*Create the device variables*/
	float *d_matrix, *d_aux_matrix, *d_weights, *d_aux_vec;

	/*Malloc the pinned memory with cuda*/
	checkCuda( cudaMallocHost((void**)&pinned_matrix, matrix_bytes));
	checkCuda( cudaMallocHost((void**)&pinned_weights, weights_bytes));

	/*Malloc the device memory*/
	checkCuda( cudaMalloc((void**)&d_matrix,      matrix_bytes));
	checkCuda( cudaMalloc((void**)&d_aux_matrix,  matrix_bytes));
	checkCuda( cudaMalloc((void**)&d_weights,     weights_bytes));
	checkCuda( cudaMalloc((void**)&d_aux_vec,     weights_bytes));

	/*Set the aux variables to 0*/
	cudaMemset(d_aux_matrix, 0, matrix_bytes);
	cudaMemset(d_aux_vec, 0, weights_bytes);

	/*Populate the matrix*/
	// The matrix is composed by each alternative resource set, for example, if exists 4 alternatives and 3 resources. The matrix is:
	/*
	   Alt1_Res1 Alt2_Res1 Alt3_Res1 Alt4_Res1
	   Alt1_Res2 Alt2_Res2 Alt3_Res2 Alt4_Res2
	   Alt1_Res3 Alt2_Res3 Alt3_Res3 Alt4_Res3
	 */

	/*Step 1 - Build the matrix*/
	// printf("Step One\n");
	{
		int i=0,j=0;
		for( i=0; i<alt_size; i++) {
			//Take advantage of this loop to populate the host index
			this->hosts_index[i]=alternatives[i]->getId();
			j=0;
			for( auto it: alternatives[i]->getResource()) {
				matrix[j*alt_size+i]= it.second;
				j++;
			}
		}
		// }
		// printf("Matrix\n");
		// for(j=0; j<resources_size; j++) {
		//      for(i=0; i<alt_size; i++) {
		//              printf("%f\t",matrix[j*alt_size+i]);
		//      }
		//      printf("\n");
		// }
	}
	// printf("Getting the weights\n");
	getWeights(weights, types, allResources);

	/*copy the values to the pinned memory*/
	memcpy(pinned_matrix, matrix, matrix_bytes);
	memcpy(pinned_weights, weights, weights_bytes);

	// printf("Free matrix\n");
	free(matrix);
	// printf("Free Weights\n");
	free(weights);
	// printf(" OK\n");
	matrix =NULL;
	weights=NULL;
	// printf("NULL OK\n");

	/*Need to free the host variables*/
	/*Copy the pinned values to the device memory*/
	checkCuda( cudaMemcpy(d_matrix, pinned_matrix, matrix_bytes, cudaMemcpyHostToDevice));
	checkCuda( cudaMemcpy(d_weights, pinned_weights, weights_bytes, cudaMemcpyHostToDevice));

	// printf("Cuda Free host\n");
	cudaFreeHost(pinned_matrix);
	// printf("PWeights\n");
	cudaFreeHost(pinned_weights);
	// printf(" OK\n");
	/*Prepare the blod and grid for kernel*/
	dim3 block_1d(block_size,1,1);
	dim3 grid_1d(ceil(alt_size/(float)block_1d.x),1,1);

	dim3 block_2d(block_size,block_size,1);
	dim3 grid_2d(ceil(alt_size/(float)block_2d.x), ceil(resources_size/(float)block_2d.y),1);

	/*Step 2 - Calculate the normalized Matrix*/
	// printf("Step Two\n");
	powKernel<<< grid_2d, block_2d>>> (d_matrix, d_aux_matrix, alt_size, resources_size);
	cudaDeviceSynchronize();

	sumLinesSqrtKernel<<< grid_1d, block_1d>>> (d_aux_matrix, d_aux_vec, alt_size, resources_size);
	cudaDeviceSynchronize();

	cudaFree(d_aux_matrix);

	/*Step 2 and 3 - Calculate the Weighted Normalized Matrix*/
	normalizeKernel<<< grid_2d, block_2d >>> (d_matrix, d_weights, d_aux_vec, alt_size, resources_size);
	cudaDeviceSynchronize();

	cudaFree(d_aux_vec);
	cudaFree(d_weights);

	/*Step 4 - Calculate the ideal best and ideal worst value*/
	float *d_min, *d_max;
	checkCuda( cudaMalloc((void**)&d_min, weights_bytes));
	checkCuda( cudaMalloc((void**)&d_max, weights_bytes));

	prepareMinMaxKernel(d_matrix, d_min, d_max, types, alt_size, resources_size);
	cudaDeviceSynchronize();

	free(types);
	// testMaxKernel();
	/*Step 5 - Calculate the euclidean distance from the ideal best and ideal worst*/

	float *max_temp_matrix, *min_temp_matrix;

	checkCuda( cudaMalloc((void**)&max_temp_matrix, matrix_bytes));
	checkCuda( cudaMalloc((void**)&min_temp_matrix, matrix_bytes));

	// cudaMemset(max_temp_matrix, 0, sizeof(float)*matrix_bytes);
	// cudaMemset(min_temp_matrix, 0, sizeof(float)*matrix_bytes);

	subtractByMinMaxKernel<<<grid_2d, block_2d>>> (d_matrix, max_temp_matrix, min_temp_matrix, d_max, d_min, alt_size, resources_size);
	cudaDeviceSynchronize();

	cudaFree(d_matrix);
	cudaFree(d_min);
	cudaFree(d_max);

	float *d_smin, *d_smax;
	checkCuda( cudaMalloc((void**)&d_smax, sizeof(float)*alt_size));
	checkCuda( cudaMalloc((void**)&d_smin, sizeof(float)*alt_size));

	cudaMemset(d_smax, 0, sizeof(float)*alt_size);
	cudaMemset(d_smin, 0, sizeof(float)*alt_size);

	ldKernel<<<grid_1d, block_1d>>>(max_temp_matrix, min_temp_matrix, d_smax, d_smin, alt_size, resources_size);
	cudaDeviceSynchronize();

	cudaFree(max_temp_matrix);
	cudaFree(min_temp_matrix);

	/*Step 6 - Calculate the similarity */

	float *d_result;
	checkCuda( cudaMalloc((void**)&d_result, result_bytes));// the result has the same length than the |alternatives|

	performanceScoreKernel<<<grid_1d, block_1d>>> (d_smax, d_smin, d_result, alt_size);
	cudaDeviceSynchronize();

	/*Step 7 - Rank the alternatives */
	float *result = (float*) malloc (result_bytes);
	checkCuda( cudaMemcpy(result, d_result, result_bytes, cudaMemcpyDeviceToHost));

	cudaFree(d_result);

	/*Store the ranked resources to future access*/
	this->hosts_value = result;
	this->hosts_size = alt_size;
	result = NULL;
}

unsigned int* TOPSIS::getResult(unsigned int& size){
	size = this->hosts_size;

	unsigned int* result = (unsigned int*) malloc (sizeof(unsigned int)*size);

	unsigned int i;

	std::priority_queue<std::pair<float, int> > alternativesPair;

	// printf("VALUES\n");
	for (i = 0; i < size; i++) {
		alternativesPair.push(std::make_pair(this->hosts_value[i], i));
		// printf("%f - ",this->hosts_value[i]);
	}
	// printf("\n");
	i=0;

	while(!alternativesPair.empty()) {
		// printf("\t%f\t%d\n",alternativesPair.top().first,alternativesPair.top().second);
		result[i] = this->hosts_index[alternativesPair.top().second];
		alternativesPair.pop();
		i++;
	}
	return result;
}

void TOPSIS::setAlternatives(Host** host, int size){
}
