#include "ahpg.cuh"

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		SPDLOG_ERROR("CUDA Runtime Error: {}", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

AHPG::AHPG() {
	this->hierarchy = NULL;
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
	this->setHierarchyG();
	this->conceptionG();
}

AHPG::~AHPG(){
	delete(hierarchy);
}

void AHPG::setHierarchyG(){
	if(this->hierarchy!=NULL) {
		delete(this->hierarchy);
	}
	this->hierarchy = new Hierarchy();
}

char* AHPG::strToLowerG(const char* str) {
	int i;
	char* res = (char*) malloc( strlen(str)+1 );
	for(i=0; i<strlen(str); i++) {
		res[i]=tolower(str[i]);
	}
	res[strlen(str)] = '\0';
	return res;
}

// Recieve a function address to iterate, used in build matrix, normalized, pml
// and pg functions.
template <typename F> void AHPG::iterateFuncG(F f, Node* node) {
	Edge** edges = node->getEdges(); // array of edges
	Node* criteria; // cruteria node
	int i;
	int size = node->getSize();
	for( i=0; i<size; i++ ) {
		criteria = edges[i]->getNode();
		if (criteria != NULL) {
			(this->*f)(criteria);
		}
	}
}

void AHPG::buildMatrixG(Node* node) {
	int i,j;
	int size = node->getSize(); // get the number of edges
	if ( size == 0 ) return;
	float* matrix = (float*) malloc (sizeof(float)* size*size);

	float* weights;

	for (i = 0; i < size; i++) {
		matrix[i*size+i] = 1;
		weights = (node->getEdges())[i]->getWeights();
		for (j = i + 1; j < size; j++) {
			if(weights == NULL ) {
				SPDLOG_ERROR("Build matrix weight is NULL");
				exit(0);
			}
			matrix[i*size+j] = weights[j];
			matrix[j*size+i] = 1.0 / matrix[i*size+j];
		}
	}

	node->setMatrix(matrix);

	iterateFuncG(&AHPG::buildMatrixG, node);
}

void AHPG::buildNormalizedMatrixG(Node* node) {
	int size = node->getSize();
	if ( size == 0 ) return;
	// //Get the device info
	int devID;
	cudaDeviceProp props;
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&props, devID);
	int block_size = (props.major < 2) ? 16 : 32;

	int bytes = size*size*sizeof(float);

	float* matrix = node->getMatrix();
	float* nMatrix = (float*) malloc (sizeof(float) * size*size);

	float* pinned_matrix;

	float* d_data, *d_sum, *d_result;

	checkCuda( cudaMallocHost((void**)&pinned_matrix, bytes));

	checkCuda( cudaMalloc((void**)&d_data, bytes));
	checkCuda( cudaMalloc((void**)&d_sum, size*sizeof(float)));
	checkCuda( cudaMalloc((void**)&d_result, bytes));

	memcpy(pinned_matrix, matrix, bytes);

	checkCuda( cudaMemcpy(d_data, pinned_matrix, bytes, cudaMemcpyHostToDevice));

	dim3 blockSum(block_size);
	dim3 gridSum(ceil(size/(float)blockSum.x));

	calculateSUM_Line<<<gridSum, blockSum>>>(d_data,d_sum, size);

	cudaDeviceSynchronize();

	dim3 blockMatrix(block_size,block_size);
	dim3 gridMatrix(ceil(size/(float)blockMatrix.x), ceil(size/(float)blockMatrix.y));

	calculateNMatrix<<< gridMatrix, blockMatrix >>>(d_data, d_sum, d_result, size);

	cudaDeviceSynchronize();
	checkCuda( cudaMemcpy(nMatrix, d_result, bytes, cudaMemcpyDeviceToHost) );
	cudaDeviceSynchronize();

	node->setNormalizedMatrix(nMatrix);
	deleteMatrixG(node);
	cudaFree(d_data);
	cudaFree(d_sum);
	cudaFree(d_result);
	cudaFreeHost(pinned_matrix);
	iterateFuncG(&AHPG::buildNormalizedMatrixG, node);
}

void AHPG::buildPmlG(Node* node) {
	int size = node->getSize();
	if ( size == 0 ) return;

	int devID;
	cudaDeviceProp props;
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&props, devID);
	int block_size = (props.major < 2) ? 16 : 32;

	float* pml = (float*) malloc (sizeof(float) * size);
	float* matrix = node->getNormalizedMatrix();

	float* pinned_matrix;

	float* d_pml, *d_data, *d_result;

	checkCuda( cudaMallocHost((void**)&pinned_matrix, sizeof(float)*size*size));

	checkCuda( cudaMalloc((void**)&d_pml, size*sizeof(float)));
	checkCuda( cudaMalloc((void**)&d_data, size*size*sizeof(float)));
	checkCuda( cudaMalloc((void**)&d_result, size*sizeof(float)));

	memcpy(pinned_matrix, matrix, sizeof(float)*size*size);

	checkCuda( cudaMemcpy(d_data, pinned_matrix, size*size*sizeof(float), cudaMemcpyHostToDevice));

	dim3 block(block_size);
	dim3 grid(ceil(size/(float)block.x));

	calculateCPml<<<grid, block>>>(d_data, d_pml, size);

	cudaDeviceSynchronize();
	checkCuda( cudaMemcpy(pml, d_pml, sizeof(float)*size, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	// spdlog::debug("INSIDE FUNCTION\n");
	// for(int i=0; i<size; i++) {
	//      spdlog::debug("%f ",pml[i]);
	// }
	// spdlog::debug("\n");
	node->setPml(pml);
	deleteNormalizedMatrixG(node);
	cudaFree(d_pml);
	cudaFree(d_data);
	cudaFree(d_result);
	cudaFreeHost(pinned_matrix);
	iterateFuncG(&AHPG::buildPmlG, node);
}

void AHPG::buildPgG(Node* node) {
	int i;
	int size = this->hierarchy->getAlternativesSize();
	if ( size == 0 ) return;
	float* pg = (float*) malloc (sizeof(float) * size);
	for (i = 0; i < size; i++) {
		pg[i] = partialPgG(node, i);
	}
	node->setPg(pg);
}

float AHPG::partialPgG(Node* node, int alternative) {
	int i;

	Node* criteria;

	Edge** edges= node->getEdges();
	int size = node->getSize();

	float* pml = node->getPml();
	float partial = 0;
	for (i = 0; i < size; i++) {
		criteria = edges[i]->getNode();
		if (criteria != NULL && criteria->getType()!=node_t::ALTERNATIVE) {
			partial += pml[i] * partialPgG(criteria, alternative);
		} else {
			return pml[alternative];
		}
	}
	return partial;
}

void AHPG::deleteMatrixG(Node* node) {
	float* matrix = node->getMatrix();
	if(matrix==NULL);
	free(matrix);
	matrix = NULL;
	node->setMatrix(NULL);
}

void AHPG::deleteNormalizedMatrixG(Node* node) {
	float* nMatrix = node->getNormalizedMatrix();
	if(nMatrix==NULL) return;
	free(nMatrix);
	nMatrix = NULL;
	node->setNormalizedMatrix(NULL);
}

void AHPG::deleteMatrixIG(Node* node) {
	float* matrix = node->getMatrix();
	if(matrix!=NULL) {
		free(matrix);
		matrix = NULL;
		node->setMatrix(NULL);
	}
	iterateFuncG(&AHPG::deleteMatrixIG, node);
}

void AHPG::deleteNormalizedMatrixIG(Node* node) {
	float* nMatrix = node->getNormalizedMatrix();
	if(nMatrix!=NULL) {
		free(nMatrix);
		nMatrix = NULL;
		node->setNormalizedMatrix(NULL);
	}
	iterateFuncG(&AHPG::deleteNormalizedMatrixIG, node);
}

void AHPG::deletePmlG(Node* node){
	float* pml = node->getPml();
	free(pml);
	pml = NULL;
	node->setPml(NULL);
	iterateFuncG(&AHPG::deletePmlG, node);
}

void AHPG::checkConsistencyG(Node* node) {
	int i, j;
	int size = node->getSize();
	float* matrix = node->getMatrix();
	float* pml = node->getPml();
	float p[size], lambda = 0, RC = 0;
	for (i = 0; i < size; i++) {
		p[i] = 0;
		for (j = 0; j < size; j++) {
			p[i] += pml[j] * matrix[i*size+j];
		}
		lambda += (p[i] / pml[i]);
	}
	lambda /= size;
	if (IR[size] > 0) {
		RC = (fabs(lambda - size) / (size - 1)) / IR[size];
	} else {
		// according to AlonsoLamata 2006
		// RC = CI/ RI , where
		// CI = (Lambda_max - n ) / (n-1), and
		// RI = (~Lambda_max - n) / (n-1) , so
		// RC = (Lambda_max - n) / (n-1) / (~Lambda_max - n) / (n-1), then
		// RC = (Lambda_max - n) / (~Lambda_max - n), the ~Lambda_max can be
		// calculated through ~Lambda_max = 2.7699*n-4.3513, thus RC = (Lambda_max -
		// n) / (2.7699 * n - 4.3513 - n ), simplifying RC = (Lambda_max - n) /
		// (1.7699 * n - 4.3513)
		RC = (abs(lambda - size) / (1.7699 * size - 4.3513));
	}
	if (RC > 0.1) {
		SPDLOG_ERROR("Criteria {} is inconsistent", node->getName());
		spdlog::debug("RC= {}", RC);
		spdlog::debug("SIZE= {}", size);
		printMatrixG(node);
		printNormalizedMatrixG(node);
		printPmlG(node);
		exit(0);
	}
	iterateFuncG(&AHPG::checkConsistencyG, node);
}

void AHPG::printMatrixG(Node* node) {
	int i,j;
	float* matrix = node->getMatrix();
	int tam = node->getSize();
	if(tam==0) return;
	spdlog::debug("Matrix of %s\n", node->getName());
	for (i = 0; i < tam; i++) {
		for (j = 0; j < tam; j++) {
			spdlog::debug("%010lf\t", matrix[i*tam+j]);
		}
		spdlog::debug("\n");
	}
	spdlog::debug("\n");
	iterateFuncG(&AHPG::printMatrixG, node);
}

void AHPG::printNormalizedMatrixG(Node* node) {
	int i,j;
	float* matrix = node->getNormalizedMatrix();
	int tam = node->getSize();
	if(tam==0) return;
	spdlog::debug("Normalized Matrix of {}", node->getName());
	for (i = 0; i < tam; i++) {
		for (j = 0; j < tam; j++) {
			spdlog::debug("{}\t", matrix[i*tam+j]);
		}
	}
	spdlog::debug("");
	iterateFuncG(&AHPG::printNormalizedMatrixG, node);
}

void AHPG::printPmlG(Node* node) {
	int i;
	float* pml = node->getPml();
	int tam = node->getSize();
	if(tam==0) return;
	spdlog::debug("PML of {}", node->getName());
	for (i = 0; i < tam; i++) {
		spdlog::debug("{}\t", pml[i]);
	}
	spdlog::debug("");
	iterateFuncG(&AHPG::printPmlG, node);
}

void AHPG::printPgG(Node* node) {
	int i;
	float* pg = node->getPg();
	int tam = this->hierarchy->getAlternativesSize();
	spdlog::debug("PG of {}", node->getName());
	for (i = 0; i < tam; i++) {
		spdlog::debug("{}\t",pg[i]);
	}
	spdlog::debug("");
}

void AHPG::hierarchyParserG(const rapidjson::Value &hierarchyData) {
	Node* focus= this->hierarchy->addFocus(hierarchyData["name"].GetString());
	Node* criteria = NULL;

	const rapidjson::Value &c_array = hierarchyData["childs"];
	size_t c_size = c_array.Size();
	float weights[c_size];

	for(size_t i=0; i< c_size; i++) {
		criteria = this->hierarchy->addCriteria(c_array[i]["name"].GetString());
		const rapidjson::Value & w_array = c_array[i]["weight"];
		for(size_t j=0; j<w_array.Size(); j++) {
			weights[j] = w_array[j].GetFloat();
		}
		this->hierarchy->addEdge(focus, criteria, weights, c_size);
	}
}

void AHPG::conceptionG() {
	// Parser the Json File that contains the Hierarchy
	char hierarchy_schema [1024] = "\0";
	char hierarchy_data [1024] = "\0";

	strcpy(hierarchy_schema, path);
	strcpy(hierarchy_data, path);

	strcat(hierarchy_schema, "multicriteria/ahp/json/hierarchySchema.json");

	if(this->type==0)
		strcat(hierarchy_data, "multicriteria/ahp/json/hierarchyData.json");
	else if(this->type==1)
		strcat(hierarchy_data, "multicriteria/ahp/json/hierarchyDataFrag.json");
	else
		SPDLOG_ERROR("Hierarchy data type error");

	rapidjson::SchemaDocument hierarchySchema =
		JSON::generateSchema(hierarchy_schema);
	rapidjson::Document hierarchyData =
		JSON::generateDocument(hierarchy_data);
	rapidjson::SchemaValidator hierarchyValidator(hierarchySchema);

	if (!hierarchyData.Accept(hierarchyValidator))
		JSON::jsonError(&hierarchyValidator);
	hierarchyParserG(hierarchyData["objective"]);
}

void AHPG::acquisitionG() {
	// //Get the device info
	int devID;
	cudaDeviceProp props;
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&props, devID);
	int block_size = (props.major < 2) ? 16 : 32;
	//
	int i,j;
	// Para gerar os pesos das alterntivas, será primeiro captado o MIN e MAX
	// valor das alternativas , após isso será montada as matrizes de cada sheet
	// auto max = this->hierarchy->getResource();
	// auto min = this->hierarchy->getResource();
	Node** alt = this->hierarchy->getAlternatives();
	Node** sheets = this->hierarchy->getCriterias();

	int altSize = this->hierarchy->getAlternativesSize();
	int sheetsSize = this->hierarchy->getCriteriasSize();
	int resourceSize = this->hierarchy->getResource()->getDataSize();

	float* min_max_values = (float*) malloc (sizeof(float) * resourceSize);
	{
		float min, max, value;

		for( i=0; i<resourceSize; i++) {
			min = FLT_MAX;
			max = FLT_MIN;
			for( j=0; j<altSize; j++ ) {
				value = alt[j]->getResource()->getResource(i);
				if( value > max) {
					max = value;
				}
				if( value < min ) {
					min = value;
				}
			}
			if(min==0 && max==1) { // the value is boolean
				min_max_values[i] = -1; //simulate boolean value
			}else{
				min_max_values[i] = (max-min)/ 9.0; // the other variables
			}
		}
	}
	// Create the data host memory
	int data_size = resourceSize*altSize;
	int result_size = sheetsSize*altSize*altSize;

	float* h_data = (float*) malloc (sizeof(float)* data_size );
	{
		// TODO CAUNTION POSSIBLE ERROR
		int index=0;
		H_Resource* temp_resource;
		for(i=0; i<altSize; i++) {
			temp_resource = alt[i]->getResource();
			for(j=0; j<temp_resource->getDataSize(); j++) {
				h_data[index] = temp_resource->getResource(j);
				index++;
			}
		}
		temp_resource = NULL;
	}

	float* pinned_data, *pinned_min_max;

	float* d_data, *d_min_max, *d_result;

	checkCuda( cudaMallocHost((void**)&pinned_data, data_size*sizeof(float)));
	checkCuda( cudaMallocHost((void**)&pinned_min_max, resourceSize*sizeof(float)));

	checkCuda( cudaMalloc((void**)&d_data, data_size*sizeof(float)));
	checkCuda( cudaMalloc((void**)&d_min_max, resourceSize*sizeof(float)));
	checkCuda( cudaMalloc((void**)&d_result, result_size*sizeof(float)));

	memcpy(pinned_data, h_data, data_size*sizeof(float));
	memcpy(pinned_min_max, min_max_values, resourceSize*sizeof(float));

	free(h_data);
	free(min_max_values);

	float* result = (float*) malloc (sizeof(float) * result_size);

	checkCuda( cudaMemcpy(d_data, pinned_data, data_size*sizeof(float), cudaMemcpyHostToDevice));
	checkCuda( cudaMemcpy(d_min_max, pinned_min_max, resourceSize*sizeof(float), cudaMemcpyHostToDevice));

	dim3 block(block_size,block_size);
	dim3 grid(ceil(altSize/(float)block.x), ceil(altSize/(float)block.y));

	// spdlog::debug("Calling the kernel\n");
	// spdlog::debug("Data size %d, result size %d\n", data_size, result_size);
	acquisitonKernel <<< grid, block >>> (d_data, d_min_max, d_result, sheetsSize, altSize);

	cudaDeviceSynchronize();
	checkCuda( cudaMemcpy(result, d_result, result_size*sizeof(float),cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	// With all the weights calculated, now the weights are set in each edge between the sheets and alternatives
	Edge** edges = NULL;
	float temp[altSize];
	for(i=0; i< sheetsSize; i++) {
		edges = sheets[i]->getEdges(); // get the array of edges' pointer
		for (j = 0; j < altSize; j++) {         // iterate trhough all the edges
			//memcpy(&temp, &result[i*sheetsSize+j*altSize], altSize);
			std::copy(result+(i*altSize*sheetsSize+j*sheetsSize),result+(i*sheetsSize*altSize+j*sheetsSize+altSize), temp);

			edges[j]->setWeights(temp, altSize);
		}
	}
	cudaFree(d_data);
	cudaFree(d_min_max);
	cudaFree(d_result);
	cudaFreeHost(pinned_data);
	cudaFreeHost(pinned_min_max);
	free(result);
}

void AHPG::synthesisG() {
	// 1 - Build the construccd the matrix
	// spdlog::debug("B M\n");
	buildMatrixG(this->hierarchy->getFocus());
	// printMatrixG(this->hierarchy->getFocus());
	// 2 - Normalize the matrix
	// spdlog::debug("B N\n");
	buildNormalizedMatrixG(this->hierarchy->getFocus());
	// printNormalizedMatrixG(this->hierarchy->getFocus());
	// spdlog::debug("B P\n");
	buildPmlG(this->hierarchy->getFocus());
	// printPmlG(this->hierarchy->getFocus());
	// 4 - calculate the PG
	// spdlog::debug("B PG\n");
	buildPgG(this->hierarchy->getFocus());
	// printPgG(this->hierarchy->getFocus());
}

void AHPG::consistencyG() {
	iterateFuncG( &AHPG::checkConsistencyG, hierarchy->getFocus() );

}

void AHPG::run(Host** alternatives, int size) {
	spdlog::debug("AHPG Init");
	this->hierarchy->clearAlternatives();
	this->hierarchy->clearResource();

	std::map<std::string, float> resource = alternatives[0]->getResource();
	spdlog::debug("Setting the alternatives values");
	for (auto it : resource) {
		this->hierarchy->addResource((char*)it.first.c_str());
	}

	this->setAlternatives(alternatives, size);
	if(this->hierarchy->getCriteriasSize()==0) {
		std::cerr<<"AHP Hierarchy with no sheets";
		exit(0);
	}

	spdlog::debug("Aquisition: ");
	this->acquisitionG();

	spdlog::debug("Synthesis");
	this->synthesisG();
	// this->consistency();
	spdlog::debug("AHPG End");

}

unsigned int* AHPG::getResult(unsigned int& size) {
	unsigned int* result = (unsigned int*) malloc (sizeof(unsigned int));

	unsigned int i;

	float* values = this->hierarchy->getFocus()->getPg();

	std::priority_queue<std::pair<float, int> > alternativesPair;

	// spdlog::debug("PG\n");
	for (i = 0; i < (unsigned int) this->hierarchy->getAlternativesSize(); i++) {

		// if( (unsigned int)alternativesPair.size()<20)
		// spdlog::debug("%f#%u$ ",values[i],i);
		alternativesPair.push(std::make_pair(values[i], i));
	}
	// spdlog::debug("\n");


	Node** alternatives = this->hierarchy->getAlternatives();

	i=0;
	while(!alternativesPair.empty()) {
		result = (unsigned int*) realloc (result, sizeof(unsigned int)*(i+1));
		result[i] = atoi(alternatives[alternativesPair.top().second]->getName());
		alternativesPair.pop();
		i++;
	}
	size = i;
	return result;
}

void AHPG::setAlternatives(Host** alternatives, int size) {
	int i;

	// this->hierarchy->clearAlternatives();

	std::map<std::string,float> resource;
	Node* a = NULL;
	for ( i=0; i<size; i++) {
		resource = alternatives[i]->getResource(); // Host resource

		a = new Node(); // create the new node

		a->setResource(this->hierarchy->getResource()); // set the default resources in the node

		a->setName((char*) std::to_string(alternatives[i]->getId()).c_str()); // set the node name

		// Update the node h_resource values by the host resource values
		for (auto it : resource) {
			a->setResource((char*)it.first.c_str(), it.second);
		}

		this->hierarchy->addAlternative(a);
	}

	this->hierarchy->addEdgeCriteriasAlternatives();
}
