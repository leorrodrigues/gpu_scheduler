#include "ahpg.cuh"

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

AHPG::AHPG() {
	IR[3] = 0.5245;
	IR[4] = 0.8815;
	IR[5] = 1.1086;
	IR[6] = 1.1279;
	IR[7] = 1.3417;
	IR[8] = 1.4056;
	IR[9] = 1.4499;
	IR[10] = 1.4854;
	IR[11] = 1.5141;
	IR[12] = 1.5365;
	IR[13] = 1.5551;
	IR[14] = 1.5713;
	IR[15] = 1.5838;
	char cwd[1024];
	char* result;
	result = getcwd(cwd, sizeof(cwd));
	if(result == NULL) {
		SPDLOG_ERROR("The directory path is not correct");
	}
	char* sub_path = (strstr(cwd, "ahp"));
	if(sub_path!=NULL) {
		int position = sub_path - cwd;
		strncpy(this->path, cwd, position);
		this->path[position-1] = '\0';
	}else{
		strcpy(this->path, cwd);
		strcat(this->path,"/");
	}

	this->hosts_size=0;

	this->d_pml_obj=NULL;
	this->pg=NULL;
}

AHPG::~AHPG(){
	checkCuda(cudaFree(this->d_pml_obj));
	this->d_pml_obj=NULL;
	this->pg=NULL;
}

void AHPG::run(Host** alternatives, int alt_size, int interval_low, int interval_high){
	int devID;
	cudaDeviceProp props;
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&props, devID);
	int block_size = (props.major <2) ? 16 : 32;

	// Creating the array for the host index
	this->hosts_size = alt_size;
	this->hosts_index = (unsigned int*) malloc (sizeof(unsigned int)* alt_size);
	// std::map<std::string, Interval_Tree::Interval_Tree*> allResources = alternatives[0]->getResource();

	//The first level of criteria was build in the edges values
	//Now its needed to build the second level of hierarchy through the alt_values array
	//The matrix is a composition of all the individuals matrix of the criteiras, this means that his size is equals to criterias_size*alternatives_size
	size_t matrix_bytes = sizeof(float)*this->criteria_size*alt_size;
	size_t values_bytes = sizeof(float)*this->criteria_size;
	// Create the host variable
	//Create the matrix that will be used in the hierarchy
	float *matrix = (float*) malloc (matrix_bytes);
	float *values = (float*) malloc (values_bytes);
	// Iterate through all the alternatives, get their ID's and their values
	{
		int i=0,j=0;
		float max[this->criteria_size];
		float min[this->criteria_size];
		float temp=0;
		for( i=0; i<this->criteria_size; i++) {
			max[i]=-1; //we don't have negative resources
			min[i]=FLT_MAX;
		}
		for( i=0; i<alt_size; i++) {
			this->hosts_index[i]=alternatives[i]->getId();
			j=0;
			for( auto it: alternatives[i]->getResource()) {
				temp = (interval_high - interval_low) * it.second->getMinValueAvailable( interval_low, interval_high);
				if(temp<0.0000000001) temp = 0;
				matrix[i+j*alt_size] = temp;
				if(max[j]<temp) max[j] = temp;
				if(min[j]>temp) min[j] = temp;
				j++;
			}
		}
		for( i=0; i<this->criteria_size; i++)
			values[i]=(max[i]-min[i])/9.0;
	}

	//With the host values inside the matrix, the create the Pinned and Cuda Variables
	// create the pinned variables
	float *pinned_matrix;
	float *pinned_values;

	// Create the device variables
	float *d_matrix;
	float *d_values;

	// Malloc the pinned memory in cuda
	checkCuda( cudaMallocHost((void**)&pinned_matrix, matrix_bytes));
	checkCuda( cudaMallocHost((void**)&pinned_values, values_bytes));

	// Malloc the device Memory
	checkCuda( cudaMalloc((void**)&d_matrix, matrix_bytes));
	checkCuda( cudaMalloc((void**)&d_values, values_bytes));

	// Copy the values to the pinned memory
	memcpy(pinned_matrix, matrix, matrix_bytes);
	memcpy(pinned_values, values, values_bytes);

	free(matrix);
	free(values);
	matrix = NULL;
	values = NULL;

	// Copy the pinned values to the device memory
	checkCuda( cudaMemcpy(d_matrix, pinned_matrix, matrix_bytes, cudaMemcpyHostToDevice));
	checkCuda( cudaMemcpy(d_values, pinned_values, values_bytes, cudaMemcpyHostToDevice));

	// Free the Pinned variables
	checkCuda(cudaFreeHost(pinned_matrix));
	checkCuda(cudaFreeHost(pinned_values));

	// Now the matrix is inside gpu memory, we need to convert their values into a range of 1 to 9, to do this, we need to get the higher and lower values from each
	{
		dim3 block(block_size, block_size,1);
		dim3 grid (ceil(alt_size/(float)block.x), ceil(this->criteria_size/(float)block.y),1);
		normalize19Kernel<<<grid,block>>>(d_matrix, d_values, alt_size, this->criteria_size);
		cudaDeviceSynchronize();
	}
	checkCuda(cudaFree(d_values));

	size_t ahp_matrix_bytes = sizeof(float)*this->criteria_size*alt_size*alt_size;
	float *d_pairwise;
	checkCuda( cudaMalloc((void**)&d_pairwise, ahp_matrix_bytes));

	// Now we make the pairwise comparison
	{
		dim3 block(block_size, block_size,1);
		dim3 grid (ceil(alt_size/(float)block.x), ceil(alt_size/(float)block.y), ceil(this->criteria_size/(float)block.z));

		// size_t smemSize = block_size * block_size * sizeof(float);

		// pairwiseComparsionKernel<<<grid,block,smemSize>>>(d_matrix, d_pairwise, alt_size, this->criteria_size);
		pairwiseComparsionKernel<<<grid,block>>>(d_matrix, d_pairwise, alt_size, this->criteria_size);
		cudaDeviceSynchronize();
	}
	checkCuda(cudaFree(d_matrix));

	// {
	//      float *temp = (float*) malloc(sizeof(float)* ahp_matrix_bytes);
	//
	//      checkCuda(cudaMemcpy(temp, d_pairwise, ahp_matrix_bytes, cudaMemcpyDeviceToHost));
	//      for(size_t i=0; i<this->criteria_size; i++) {
	//              for(size_t j=0; j<alt_size; j++) {
	//                      for(size_t k=0; k<alt_size; k++) {
	//                              std::cout<<temp[i*alt_size*alt_size+j*alt_size+k]<<" ";
	//                              if(temp[i*alt_size*alt_size+j*alt_size+k]==0) {
	//                                      printf("Erro in the cuda acquisition\n");
	//                                      free(temp);
	//                                      exit(0);
	//                              }
	//                      }
	//                      std::cout<<"\n";
	//              }
	//              std::cout<<"\n\n\n";
	//      }
	//      free(temp);
	// }

	// NORMALIZE MATRIX
	float *d_sum;
	size_t sum_bytes = sizeof(float)*this->criteria_size*alt_size;
	checkCuda( cudaMalloc((void**)&d_sum, sum_bytes));
	//The pairwise matrix is composed by Criterias X Alternatives X Alternatives
	//To normalize the matrix, first we need the sum of each line in the pairwise matrix (remember that the pairwise is a 3D matrix, so we have a matrix os sum lines)
	{
		{
			dim3 block(block_size, block_size, 1);
			dim3 grid (ceil(alt_size/(float)block.x), ceil(this->criteria_size/(float)block.y),1);

			sumRowKernel<<<grid,block>>>(d_pairwise, d_sum, alt_size, this->criteria_size);

			cudaDeviceSynchronize();
		}

		//With the sum array made, we make a normalization on the pairwise matrix
		{
			dim3 block(block_size, block_size, 1);
			dim3 grid (ceil(alt_size/(float)block.x), ceil(alt_size/(float)block.y),ceil(this->criteria_size)/(float)block.z);

			normalizeMatrixKernel<<<grid,block>>>(d_pairwise, d_sum, alt_size, this->criteria_size);

			cudaDeviceSynchronize();
		}
	}

	// {
	//      float *temp = (float*)malloc(sum_bytes);
	//      float *temp_p = (float*)malloc(ahp_matrix_bytes);
	//      checkCuda(cudaMemcpy(temp, d_sum, sum_bytes, cudaMemcpyDeviceToHost));
	//      checkCuda(cudaMemcpy(temp_p, d_pairwise, ahp_matrix_bytes, cudaMemcpyDeviceToHost));
	//
	//      printf("Normalize Matrix\n");
	//      for (size_t i=0; i<this->criteria_size; i++) {
	//              for(size_t j=0; j<alt_size; j++) {
	//                      for(size_t k=0; k<alt_size; k++) {
	//                              std::cout<<temp_p[i*alt_size*alt_size+j*alt_size+k]<<" ";
	//                      }
	//                      std::cout<<"\n";
	//              }
	//              std::cout<<"\n";
	//              std::cout<<"\n";
	//      }
	//      printf("Sum Matrix\n");
	//      for(size_t i=0; i<this->criteria_size; i++) {
	//              for(size_t j=0; j<alt_size; j++) {
	//                      std::cout<<temp[i*alt_size+j]<<" ";
	//              }
	//              printf("\n");
	//      }
	// printf("\n\n\n");
	// }

	// Now the pairwise matrix is normalized.
	// Generate the PML of each L1 criteria
	float *d_pml;
	{
		{
			dim3 block(block_size, block_size,1);
			dim3 grid   (ceil(alt_size/(float)block.x),ceil(this->criteria_size/(float)block.y),1);

			//reuse the SUM used in the last kernel call to this step, reducing the time by don't call again cudaMalloc and cudaFree
			sumColumnKernel<<<grid,block>>>(d_pairwise, d_sum, alt_size, this->criteria_size);

			cudaDeviceSynchronize();
			checkCuda(cudaFree(d_pairwise));
		}
		{
			size_t pml_bytes = sizeof(float)*this->criteria_size*alt_size;
			checkCuda( cudaMalloc((void**)&d_pml, pml_bytes));

			dim3 block(block_size, block_size,1);
			dim3 grid (ceil(alt_size/(float)block.x), ceil(alt_size/(float)block.y),ceil(this->criteria_size/(float)block.z));

			pmlKernel<<<grid,block>>>(d_sum, d_pml, alt_size, this->criteria_size);

			cudaDeviceSynchronize();
			checkCuda(cudaFree(d_sum));
		}
	}
	// {
	//      float *temp_p = (float*) malloc(sizeof(float)*alt_size*this->criteria_size);
	//      checkCuda(cudaMemcpy(temp_p,d_pml,(sizeof(float)*alt_size*this->criteria_size), cudaMemcpyDeviceToHost));
	//
	//      printf("PML\n");
	//      for(size_t i=0; i<this->criteria_size; i++) {
	//              for(size_t j=0; j<alt_size; j++) {
	//                      std::cout<<temp_p[i*alt_size+j]<<" ";
	//              }
	//              printf("\n");
	//      }
	//     printf("\n\n\n");
	// }

	// Now the PML were made, generate the PG
	float *d_pg;
	size_t pg_bytes = sizeof(float)*alt_size;
	checkCuda( cudaMalloc((void**)&d_pg, pg_bytes));
	{
		//PG_A1 = (PML_OBJ_C1 * PML_C1_A1 + PML_OBJ_C2 * PML_C2_A1 ... + PML_OBJ_Cn * PML_Cn * A1)
		dim3 block(block_size,1,1);
		dim3 grid   (ceil(alt_size/(float)block.x), 1,1);

		pgKernel<<<grid,block>>>(d_pml_obj, d_pml,d_pg, alt_size, this->criteria_size);
		cudaDeviceSynchronize();
	}
	checkCuda(cudaFree(d_pml));
	// With the PG calculated, update the HOST

	if(this->pg!=NULL)
		free(pg);
	this->pg = (float*)malloc(pg_bytes);

	checkCuda(cudaMemcpy(this->pg, d_pg, pg_bytes, cudaMemcpyDeviceToHost));
	checkCuda(cudaFree(d_pg));

	{
		// printf("PG\n");
		// for(int i=0; i<alt_size; i++) {
		//      std::cout<<pg[i]<<" ";
		// }
		// printf("\n");
	}
}

unsigned int* AHPG::getResult(unsigned int& size){
	size = this->hosts_size;

	unsigned int* result = (unsigned int*) malloc (sizeof(unsigned int)*size);

	unsigned int i;

	std::priority_queue<std::pair<float, int> > alternativesPair;

	for (i = 0; i < (unsigned int) this->hosts_size; i++) {
		alternativesPair.push(std::make_pair(this->pg[i], i));
	}
	i=0;

	while(!alternativesPair.empty()) {
		result[i] = this->hosts_index[alternativesPair.top().second];
		alternativesPair.pop();
		i++;
	};

	// for(int i=0; i< result)
	return result;
}

void AHPG::readJson() {
	// Parser the Json File that contains the Hierarchy
	char hierarchy_schema [1024] = "\0";
	char hierarchy_data [1024] = "\0";

	strcpy(hierarchy_schema, this->path);
	strcpy(hierarchy_data, this->path);

	strcat(hierarchy_schema, "ahp/hierarchySchema.json");

	if(this->type==0)
		strcat(hierarchy_data, "ahp/hierarchyData.json");
	else if(this->type==1)
		strcat(hierarchy_data, "ahp/hierarchyDataFrag.json");
	else if(this->type==2)
		strcat(hierarchy_data, "ahp/hierarchyDataBW.json");
	else
		SPDLOG_ERROR("Hierarchy data type error");

	rapidjson::SchemaDocument hierarchySchema =
		JSON::generateSchema(hierarchy_schema);
	rapidjson::Document hierarchyData =
		JSON::generateDocument(hierarchy_data);
	rapidjson::SchemaValidator hierarchyValidator(hierarchySchema);

	if (!hierarchyData.Accept(hierarchyValidator))
		JSON::jsonError(&hierarchyValidator);
	parseAHPG(hierarchyData["objective"]);
}

void AHPG::parseAHPG(const rapidjson::Value &hierarchyData){
	int devID;
	cudaDeviceProp props;
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&props, devID);
	int block_size = (props.major <2) ? 16 : 32;

	// the hierarchy analysed is composed by just 1 level, so the criteria_size will be the criterias + 1 (the objective/focus node).
	const rapidjson::Value &c_array = hierarchyData["childs"];
	this->criteria_size = c_array.Size();

	dim3 block_1d(block_size, 1, 1);
	dim3 grid_1d   (ceil(this->criteria_size/(float)block_1d.x),  1, 1);

	dim3 block_2d(block_size, block_size, 1);
	dim3 grid_2d   (ceil(this->criteria_size/(float)block_2d.x), ceil(this->criteria_size/(float)block_2d.y),1);

	// the edge values has the power of 2 of the criterias ammount due to the pairwise comparison.
	float *d_pairwise;
	{
		size_t pair_bytes = sizeof(float)*this->criteria_size*this->criteria_size;
		float *edges_values = (float*) malloc (pair_bytes);

		// iterate through the edges values
		// The array represents a Square Matrix
		for(size_t i=0; i< this->criteria_size; i++) {
			const rapidjson::Value & w_array = c_array[i]["weight"];
			for(size_t j=0; j<w_array.Size(); j++) {
				//according to the json, the edge_values will be composed by
				// [[w1c1 w1c2 w1c3 w1c4]
				//  [w2c1 w2c2 w3c3 w2c4]
				//  [w3c1 w3c2 w3c3 w3c4]
				//  [w4c1 w4c2 w4c3 w4c4]]

				edges_values[i*this->criteria_size+j] = w_array[j].GetFloat();
			}
		}
		// Now the edge has the pairwise comparison between the Objective and L1 Criterias. Copy the values to the GPU
		checkCuda(cudaMalloc((void**)&d_pairwise, pair_bytes));
		checkCuda(cudaMemcpy(d_pairwise, edges_values, pair_bytes, cudaMemcpyHostToDevice));
		free(edges_values);
	}
	// Making the normalized matrix
	float *d_sum;
	size_t sum_bytes = sizeof(float)*this->criteria_size;
	checkCuda( cudaMalloc((void**)&d_sum, sum_bytes));
	{
		{
			sumRowKernel<<<grid_1d,block_1d>>>(d_pairwise, d_sum, this->criteria_size,1);

			cudaDeviceSynchronize();
		}
		{
			normalizeMatrixKernel<<<grid_2d,block_2d>>>(d_pairwise, d_sum, this->criteria_size, 1);

			cudaDeviceSynchronize();
		}
	}
	// Making the pml;
	{
		{
			sumColumnKernel<<<grid_1d,block_1d>>>(d_pairwise, d_sum, this->criteria_size, 1);

			cudaDeviceSynchronize();
			checkCuda(cudaFree(d_pairwise));
		}
		{
			checkCuda(cudaMalloc((void**)&this->d_pml_obj, sizeof(float)*this->criteria_size));

			pmlKernel<<<grid_2d,block_2d>>>(d_sum, d_pml_obj, this->criteria_size, 1);

			checkCuda(cudaFree(d_sum));
		}
	}
}

void AHPG::setAlternatives(Host** alternatives, int size, int low, int high){
}
