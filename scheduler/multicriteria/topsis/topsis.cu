#include "topsis.cuh"

inline
cudaError_t checkCuda(cudaError_t result){
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		SPDLOG_ERROR("CUDA Runtime Error: {}", cudaGetErrorString(result));
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
		spdlog::debug("TOPSIS Error get directory path");
	}
	char* sub_path = (strstr(cwd, "topsis"));
	if(sub_path!=NULL) {
		int position = sub_path - cwd;
		strncpy(this->path, cwd, position);
		strcat(this->path,"/");
		this->path[position] = '\0';
	}else{
		strcpy(this->path, cwd);
		strcat(this->path,"/\0");
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

void TOPSIS::getWeights(float* weights, unsigned int* types, std::map<std::string, Interval_Tree::Interval_Tree*> resource){
	char weights_schema_path [1024] = "\0";
	char weights_data_path [1024] = "\0";

	strcpy(weights_schema_path, path);
	strcpy(weights_data_path, path);

	strcat(weights_schema_path, "topsis/weightsSchema.json");

	if(this->type==0)
		strcat(weights_data_path, "topsis/weightsData.json");
	else if(this->type==1)
		strcat(weights_data_path, "topsis/weightsDataFrag.json");
	else if(this->type==2)
		strcat(weights_data_path, "topsis/weightsDataBW.json");
	else
		SPDLOG_ERROR("Weights data type error");

	rapidjson::SchemaDocument weightsSchema = JSON::generateSchema(weights_schema_path);
	rapidjson::Document weightsData = JSON::generateDocument(weights_data_path);
	rapidjson::SchemaValidator weightsValidator(weightsSchema);
	if(!weightsData.Accept(weightsValidator))
		JSON::jsonError(&weightsValidator);

	int index=0;

	const rapidjson::Value &w_array = weightsData["weights"];
	for(size_t i=0; i<w_array.Size(); i++) {
		index = distance(resource.begin(), resource.find(w_array[i]["name"].GetString()));
		weights[index] = w_array[i]["value"].GetFloat();
		types[index] = (w_array[i]["prop"].GetBool()) ? 1 : 0;
	}
}

void TOPSIS::run(Host** alternatives, int alt_size, int interval_low, int interval_high){
	// spdlog::debug("Running topsis");
	int devID;
	cudaDeviceProp props;
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&props, devID);
	int block_size = (props.major <2) ? 16 : 32;

	this->hosts_index = (unsigned int*) malloc (sizeof(unsigned int)* alt_size);

	std::map<std::string, Interval_Tree::Interval_Tree*> allResources = alternatives[0]->getResource();

	int resources_size = allResources.size();

	size_t matrix_bytes  = sizeof(float)*resources_size*alt_size;
	size_t weights_bytes = sizeof(float)*(resources_size+1);
	size_t result_bytes  = sizeof(float)*alt_size;

	/*Create the host variables*/
	float *matrix = (float*) malloc (matrix_bytes);
	float *weights= (float*) malloc (weights_bytes);
	unsigned int * types = (unsigned int*) malloc (sizeof(unsigned int)*resources_size);
	// spdlog::debug("Allocating %d spaces for weights", resources_size);
	std::map<std::string, Interval_Tree::Interval_Tree*> a = allResources;
	// for(std::map<std::string, float>::iterator it=a.begin(); it!=a.end(); it++) {
	//      spdlog::debug("%s %f",it->first.c_str(),it->second);
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
	// spdlog::debug("Step One");
	{
		int i=0,j=0;
		float temp_capacity = 0;
		for( i=0; i<alt_size; i++) {
			//Take advantage of this loop to populate the host index
			this->hosts_index[i]=alternatives[i]->getId();
			j=0;
			for( auto it: alternatives[i]->getResource()) {
				temp_capacity = (interval_high - interval_low) * it.second->getMinValueAvailable(interval_low, interval_high);
				matrix[j*alt_size+i]= temp_capacity;
				j++;
			}
		}
		// }
		// spdlog::debug("Matrix");
		// for(j=0; j<resources_size; j++) {
		//      for(i=0; i<alt_size; i++) {
		//              spdlog::debug("%f\t",matrix[j*alt_size+i]);
		//      }
		//      spdlog::debug("");
		// }
	}
	// spdlog::debug("Getting the weights");
	getWeights(weights, types, allResources);
	/*copy the values to the pinned memory*/
	memcpy(pinned_matrix, matrix, matrix_bytes);
	memcpy(pinned_weights, weights, weights_bytes);

	// spdlog::debug("Free matrix");
	free(matrix);
	// spdlog::debug("Free Weights");
	free(weights);
	// spdlog::debug(" OK");
	matrix =NULL;
	weights=NULL;
	// spdlog::debug("NULL OK");
	/*Need to free the host variables*/
	/*Copy the pinned values to the device memory*/
	checkCuda( cudaMemcpy(d_matrix, pinned_matrix, matrix_bytes, cudaMemcpyHostToDevice));
	checkCuda( cudaMemcpy(d_weights, pinned_weights, weights_bytes, cudaMemcpyHostToDevice));

	checkCuda(cudaFreeHost(pinned_matrix));
	checkCuda(cudaFreeHost(pinned_weights));
	/*Prepare the blod and grid for kernel*/
	dim3 block_1d(block_size,1,1);
	dim3 grid_1d(ceil(alt_size/(float)block_1d.x),1,1);

	dim3 block_2d(block_size,block_size,1);
	dim3 grid_2d(ceil(alt_size/(float)block_2d.x), ceil(resources_size/(float)block_2d.y),1);

	/*Step 2 - Calculate the normalized Matrix*/
	// spdlog::debug("Step Two");
	powKernel<<< grid_2d, block_2d>>> (d_matrix, d_aux_matrix, alt_size, resources_size);
	cudaDeviceSynchronize();

	sumLinesSqrtKernel<<< grid_1d, block_1d>>> (d_aux_matrix, d_aux_vec, alt_size, resources_size);
	cudaDeviceSynchronize();

	checkCuda(cudaFree(d_aux_matrix));

	/*Step 2 and 3 - Calculate the Weighted Normalized Matrix*/
	normalizeKernel<<< grid_2d, block_2d >>> (d_matrix, d_weights, d_aux_vec, alt_size, resources_size);
	cudaDeviceSynchronize();

	checkCuda(cudaFree(d_aux_vec));
	checkCuda(cudaFree(d_weights));

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

	checkCuda(cudaFree(d_matrix));
	checkCuda(cudaFree(d_min));
	checkCuda(cudaFree(d_max));

	float *d_smin, *d_smax;
	checkCuda( cudaMalloc((void**)&d_smax, sizeof(float)*alt_size));
	checkCuda( cudaMalloc((void**)&d_smin, sizeof(float)*alt_size));

	cudaMemset(d_smax, 0, sizeof(float)*alt_size);
	cudaMemset(d_smin, 0, sizeof(float)*alt_size);

	ldKernel<<<grid_1d, block_1d>>>(max_temp_matrix, min_temp_matrix, d_smax, d_smin, alt_size, resources_size);
	cudaDeviceSynchronize();

	checkCuda(cudaFree(max_temp_matrix));
	checkCuda(cudaFree(min_temp_matrix));

	/*Step 6 - Calculate the similarity */

	float *d_result;
	checkCuda( cudaMalloc((void**)&d_result, result_bytes));// the result has the same length than the |alternatives|

	performanceScoreKernel<<<grid_1d, block_1d>>> (d_smax, d_smin, d_result, alt_size);
	cudaDeviceSynchronize();

	/*Step 7 - Rank the alternatives */
	float *result = (float*) malloc (result_bytes);
	checkCuda( cudaMemcpy(result, d_result, result_bytes, cudaMemcpyDeviceToHost));

	checkCuda(cudaFree(d_result));

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

	// spdlog::debug("VALUES");
	for (i = 0; i < size; i++) {
		alternativesPair.push(std::make_pair(this->hosts_value[i], i));
		// spdlog::debug("%f - ",this->hosts_value[i]);
	}
	// spdlog::debug("");
	i=0;

	while(!alternativesPair.empty()) {
		// spdlog::debug("\t%f\t%d",alternativesPair.top().first,alternativesPair.top().second);
		result[i] = this->hosts_index[alternativesPair.top().second];
		alternativesPair.pop();
		i++;
	}
	return result;
}

void TOPSIS::setAlternatives(Host** alternatives, int size, int low, int high){
}

void TOPSIS::readJson(){
}
