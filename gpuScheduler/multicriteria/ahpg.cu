#include "ahpg.cuh"

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
		printf("AHP Error get directory path\n");
	}
	char* sub_path = (strstr(cwd, "multicriteria"));
	if(sub_path!=NULL) {
		int position = sub_path - cwd;
		strncpy(this->path, cwd, position);
		this->path[position-1] = '\0';
	}else{
		strcpy(this->path, cwd);
		strcat(this->path,"/");
	}
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

void AHPG::updateAlternativesG() {

	char alt_schema_path [1024];
	char alt_data_path [1024];

	strcpy(alt_schema_path, path);
	strcpy(alt_data_path, path);

	strcat(alt_schema_path, "multicriteria/json/alternativesSchema.json");
	strcat(alt_data_path, "multicriteria/json/alternativesDataDefault.json");

	this->hierarchy->clearAlternatives();
	rapidjson::SchemaDocument alternativesSchema =
		JSON::generateSchema(alt_schema_path);
	rapidjson::Document alternativesData =
		JSON::generateDocument(alt_data_path);
	rapidjson::SchemaValidator alternativesValidator(alternativesSchema);
	if (!alternativesData.Accept(alternativesValidator))
		JSON::jsonError(&alternativesValidator);
	// domParser(&alternativesData, this);
	this->hierarchy->addEdgeSheetsAlternatives();
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
				printf("ahpg.cu(120) Error build matrix weight=NULL\n");
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

	calculateSUM_Line<<<blockSum, gridSum>>>(d_data,d_sum, size);

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

	calculateCPml<<<block, grid>>>(d_data, d_pml, size);

	cudaDeviceSynchronize();
	checkCuda( cudaMemcpy(pml, d_pml, sizeof(float)*size, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	// printf("INSIDE FUNCTION\n");
	// for(int i=0; i<size; i++) {
	//      printf("%f ",pml[i]);
	// }
	// printf("\n");
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
		printf("ERROR: Criteria %s is inconsistent\n", node->getName());
		printf("RC= %lf\n", RC);
		printf("SIZE= %d\n", size);
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
	printf("Matrix of %s\n", node->getName());
	for (i = 0; i < tam; i++) {
		for (j = 0; j < tam; j++) {
			printf("%010lf\t", matrix[i*tam+j]);
		}
		printf("\n");
	}
	printf("\n");
	iterateFuncG(&AHPG::printMatrixG, node);
}

void AHPG::printNormalizedMatrixG(Node* node) {
	int i,j;
	float* matrix = node->getNormalizedMatrix();
	int tam = node->getSize();
	if(tam==0) return;
	printf("Normalized Matrix of %s\n", node->getName());
	for (i = 0; i < tam; i++) {
		for (j = 0; j < tam; j++) {
			printf("%010lf\t", matrix[i*tam+j]);
		}
		printf("\n");
	}
	std::cout << "\n";
	iterateFuncG(&AHPG::printNormalizedMatrixG, node);
}

void AHPG::printPmlG(Node* node) {
	int i;
	float* pml = node->getPml();
	int tam = node->getSize();
	if(tam==0) return;
	printf("PML of %s\n", node->getName());
	for (i = 0; i < tam; i++) {
		printf("%010lf\t", pml[i]);
	}
	printf("\n");
	iterateFuncG(&AHPG::printPmlG, node);
}

void AHPG::printPgG(Node* node) {
	int i;
	float* pg = node->getPg();
	int tam = this->hierarchy->getAlternativesSize();
	printf("PG of %s\n", node->getName());
	for (i = 0; i < tam; i++) {
		printf("%010lf\t",pg[i]);
	}
	printf("\n");
}

void AHPG::generateContentSchemaG() {
	std::string names;
	std::string text = "{\"$schema\":\"http://json-schema.org/draft-04/"
	                   "schema#\",\"definitions\": {\"alternative\": {\"type\": "
	                   "\"array\",\"minItems\": 1,\"items\":{\"properties\": {";
	H_Resource* resource = this->hierarchy->getResource();
	int i, size = resource->getDataSize();
	for (i=0; i<size; i++) {
		text += "\"" + std::to_string(resource->getResource(i)) + "\":{\"type\":\"number\"},";
		names += "\"" + std::string(resource->getResourceName(i)) + "\",";
	}
	names.pop_back();
	text.pop_back();
	text += "},\"additionalProperties\": false,\"required\": [" + names +
	        "]}}},\"type\": \"object\",\"minProperties\": "
	        "1,\"additionalProperties\": false,\"properties\": "
	        "{\"alternatives\": {\"$ref\": \"#/definitions/alternative\"}}}";
	JSON::writeJson("multicriteria/json/alternativesSchema.json", text);
}

void AHPG::resourcesParserG(genericValue* dataResource) {
	std::string variableName, variableType;
	for (auto &arrayData : dataResource->value.GetArray()) {
		variableName = variableType = "";
		for (auto &objectData : arrayData.GetObject()) {
			if (strcmp(objectData.name.GetString(), "name") == 0) {
				variableName = objectData.value.GetString();
			} else if (strcmp(objectData.name.GetString(), "variableType") == 0) {
				variableType = strToLowerG(objectData.value.GetString());
			} else {
				std::cout << "Error in reading resources\nExiting...\n";
				exit(0);
			}
		}
		this->hierarchy->addResource((char*)variableName.c_str());
	}
}

void AHPG::hierarchyParserG(genericValue* dataObjective) {
	char* name = NULL;
	for (auto &hierarchyObject : dataObjective->value.GetObject()) {
		if (strcmp(hierarchyObject.name.GetString(), "name") == 0) {
			name = strToLowerG(hierarchyObject.value.GetString());
			this->hierarchy->addFocus(name); // create the Focus* in the hierarchy;
			free ( name );
		} else if (strcmp(hierarchyObject.name.GetString(), "childs") == 0) {
			criteriasParserG(&hierarchyObject, this->hierarchy->getFocus() );
		} else {
			std::cout << "AHP -> Unrecognizable Type\nExiting...\n";
			exit(0);
		}
	}
}

void AHPG::criteriasParserG(genericValue* dataCriteria, Node* parent) {
	char* name=(char*)"\0";
	bool leaf = false;
	float* weight = NULL;
	int index=0;
	for (auto &childArray : dataCriteria->value.GetArray()) {
		weight = NULL;
		index=0;
		for (auto &child : childArray.GetObject()) {
			const char* n = child.name.GetString();
			if (strcmp(n, "name") == 0) {
				name = strToLowerG(child.value.GetString());
			} else if (strcmp(n, "leaf") == 0) {
				leaf = child.value.GetBool();
			} else if (strcmp(n, "weight") == 0) {
				for (auto &weightChild : child.value.GetArray()) {
					weight = (float*) realloc (weight, sizeof(float) * (index+1) );
					weight[index]=weightChild.GetFloat();
					index++;
				}
			} else if (strcmp(n, "childs") == 0) {
				// at this point, all the criteria variables were read, now the document
				// has the child's of the criteria. To put the childs corretly inside
				// the hierarchy, the criteria node has to be created.
				auto criteria = this->hierarchy->addCriteria(name);
				criteria->setLeaf(leaf);
				this->hierarchy->addEdge(parent, criteria, weight, index);
				// with the criteria node added, the call recursively the
				// criteriasParser.
				criteriasParserG(&child, criteria);
				free(name);
			}
		}
		if (leaf) {
			auto criteria = this->hierarchy->addCriteria(name);
			criteria->setLeaf(leaf);
			this->hierarchy->addSheets(criteria);
			this->hierarchy->addEdge(parent, criteria, weight, index);
		}
		free(name);
		free(weight);
	}
}

void AHPG::alternativesParserG(genericValue* dataAlternative) {
	for (auto &arrayAlternative : dataAlternative->value.GetArray()) {
		auto alternative = this->hierarchy->addAlternative();
		for (auto &alt : arrayAlternative.GetObject()) {
			std::string name(alt.name.GetString());
			if (alt.value.IsNumber()) {
				alternative->getResource()->addResource((char*) name.c_str(), alt.value.GetFloat());
			} else if (alt.value.IsBool()) {
				bool b = alt.value.GetBool();
				if(b==true) {
					alternative->getResource()->addResource((char*) name.c_str(), 1);
				}else{
					alternative->getResource()->addResource((char*) name.c_str(), 0);
				}
			} else {
				if(name=="name") {
					alternative->setName(strToLowerG(alt.value.GetString()));
				}
			}
		}
	}
}

void AHPG::domParserG(rapidjson::Document *data) {
	for (auto &m : data->GetObject()) { // query through all objects in data.
		if (strcmp(m.name.GetString(), "resources") == 0) {
			resourcesParserG(&m);
		} else if (strcmp(m.name.GetString(), "objective") == 0) {
			hierarchyParserG(&m);
		} else if (strcmp(m.name.GetString(), "alternatives") == 0) {
			alternativesParserG(&m);
		}
	}
}

void AHPG::conceptionG(bool alternativeParser) {
	// The hierarchy contruction were divided in three parts, first the resources
	// file was to be loaded to construct the alternatives dynamically. Second the
	// hierarchy focus and criteria were loaded in the hierarchyData.json, and
	// finally the alternatives were loaded.
	if (alternativeParser) {
		char resource_schema[1024];
		char resource_data[1024];
		strcpy(resource_schema, path);
		strcpy(resource_data, path);
		strcat(resource_schema, "multicriteria/json/resourcesSchema.json");
		strcat(resource_data, "multicriteria/json/resourcesData.json");

		rapidjson::SchemaDocument resourcesSchema =
			JSON::generateSchema(resource_schema);
		rapidjson::Document resourcesData =
			JSON::generateDocument(resource_data);
		rapidjson::SchemaValidator resourcesValidator(resourcesSchema);
		if (!resourcesData.Accept(resourcesValidator))
			JSON::jsonError(&resourcesValidator);
		domParserG(&resourcesData);
		generateContentSchemaG();
	}
	// After reading the resoucesData, new alternativesSchema has to be created.
	// Parser the Json File that contains the Hierarchy
	char hierarchy_schema [1024] = "\0";
	char hierarchy_data [1024] = "\0";

	strcpy(hierarchy_schema, path);
	strcpy(hierarchy_data, path);

	strcat(hierarchy_schema, "multicriteria/json/hierarchySchema.json");
	strcat(hierarchy_data, "multicriteria/json/hierarchyDataFrag.json");

	rapidjson::SchemaDocument hierarchySchema =
		JSON::generateSchema(hierarchy_schema);
	rapidjson::Document hierarchyData =
		JSON::generateDocument(hierarchy_data);
	rapidjson::SchemaValidator hierarchyValidator(hierarchySchema);

	if (!hierarchyData.Accept(hierarchyValidator))
		JSON::jsonError(&hierarchyValidator);
	domParserG(&hierarchyData);

	if (alternativeParser) {
		char alternative_schema [1024];
		char alternative_data [1024];

		strcpy(alternative_schema, path);
		strcpy(alternative_data, path);

		strcat(alternative_schema, "multicriteria/json/alternativesSchema.json");
		strcat(alternative_data, "multicriteria/json/alternativesDataDefault.json");
		// The Json Data is valid and can be used to construct the hierarchy.
		rapidjson::SchemaDocument alternativesSchema =
			JSON::generateSchema(alternative_schema);
		rapidjson::Document alternativesData = JSON::generateDocument(alternative_data);
		rapidjson::SchemaValidator alternativesValidator(alternativesSchema);
		if (!alternativesData.Accept(alternativesValidator))
			JSON::jsonError(&alternativesValidator);
		domParserG(&alternativesData);
		this->hierarchy->addEdgeSheetsAlternatives();
	}
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
	Node** sheets = this->hierarchy->getSheets();

	int altSize = this->hierarchy->getAlternativesSize();
	int sheetsSize = this->hierarchy->getSheetsSize();
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

	// printf("Calling the kernel\n");
	// printf("Data size %d, result size %d\n", data_size, result_size);
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
	// printf("B M\n");
	buildMatrixG(this->hierarchy->getFocus());
	// printMatrixG(this->hierarchy->getFocus());
	// 2 - Normalize the matrix
	// printf("B N\n");
	buildNormalizedMatrixG(this->hierarchy->getFocus());
	// printNormalizedMatrixG(this->hierarchy->getFocus());
	// printf("B P\n");
	buildPmlG(this->hierarchy->getFocus());
	// printPmlG(this->hierarchy->getFocus());
	// 4 - calculate the PG
	// printf("B PG\n");
	buildPgG(this->hierarchy->getFocus());
	// printPgG(this->hierarchy->getFocus());
}

void AHPG::consistencyG() {
	iterateFuncG( &AHPG::checkConsistencyG, hierarchy->getFocus() );

}

void AHPG::run(Host** alternatives, int size) {
	this->setHierarchyG();
	if (size == 0) {
		this->conceptionG(true);
	} else {
		// this->hierarchy->clearAlternatives(); // made in the setAlternatives function
		this->hierarchy->clearResource();

		std::map<std::string, float> resource = alternatives[0]->getResource();

		for (auto it : resource) {
			this->hierarchy->addResource((char*)it.first.c_str());
		}

		// printf("Conception\n");
		this->conceptionG(false);
		// Add the resource of how many virtual resources are allocated in the host
		this->hierarchy->addResource((char*)"allocated_resources");
		this->setAlternatives(alternatives, size);
		if(this->hierarchy->getSheetsSize()==0) exit(0);
	}
	// printf("Aquisition: ");
	this->acquisitionG();
	// printf("Synthesis\n");
	this->synthesisG();
	// this->consistency();
}

unsigned int* AHPG::getResult(unsigned int& size) {
	unsigned int* result = (unsigned int*) malloc (sizeof(unsigned int));

	float* values = this->hierarchy->getFocus()->getPg();
	std::priority_queue<std::pair<float, int> > alternativesPair;

	unsigned int i;

	for (i = 0; i < (unsigned int) this->hierarchy->getAlternativesSize(); i++) {
		alternativesPair.push(std::make_pair(values[i], i));
	}

	size = this->hierarchy->getAlternativesSize();

	Node** alternatives = this->hierarchy->getAlternatives();

	for (i = 0; i < (unsigned int)alternativesPair.size(); i++) {
		result = (unsigned int*) realloc (result, sizeof(unsigned int)*(i+1));
		result[i] = atoi(alternatives[alternativesPair.top().second]->getName());
		alternativesPair.pop();
	}

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

		// Populate the alternative node with the Host Value
		a->setResource((char*)"allocated_resources", alternatives[i]->getAllocatedResources());

		this->hierarchy->addAlternative(a);
	}

	this->hierarchy->addEdgeSheetsAlternatives();
}
