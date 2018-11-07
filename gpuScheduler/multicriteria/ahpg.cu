#include "ahpg.cuh"

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
}

AHPG::~AHPG(){
	delete(hierarchy);
}

void AHPG::setHierarchyG(){
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
	this->hierarchy->clearAlternatives();
	rapidjson::SchemaDocument alternativesSchema =
		JSON::generateSchema("multicriteria/json/alternativesSchema.json");
	rapidjson::Document alternativesData =
		JSON::generateDocument("multicriteria/json/alternativesDataDefault.json");
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
	float** matrix = (float**) malloc (sizeof(float*) * size);
	for (i = 0; i < size; i++)
		matrix[i] = (float*) malloc (sizeof(float) * size);

	float* weights;

	for (i = 0; i < size; i++) {
		matrix[i][i] = 1;
		weights = (node->getEdges())[i]->getWeights();
		for (j = i + 1; j < size; j++) {
			matrix[i][j] = weights[j];
			matrix[j][i] = 1 / matrix[i][j];
		}
	}

	node->setMatrix(matrix);

	for(i=0; i< size; i++)
		free(matrix[i]);
	free(matrix);

	iterateFuncG(&AHPG::buildMatrixG, node);
}

void AHPG::buildNormalizedmatrixG(Node* node) {
	int i,j;
	int size = node->getSize();
	if ( size == 0 ) return;
	float** matrix = node->getMatrix(), sum = 0;
	float** nMatrix = (float**) malloc (sizeof(float*) * size);
	for (i = 0; i < size; i++)
		nMatrix[i] = (float*) malloc (sizeof(float) * size);
	for (i = 0; i < size; i++) {
		sum = 0;
		for (j = 0; j < size; j++) {
			sum += matrix[j][i];
		}
		for (j = 0; j < size; j++) {
			nMatrix[j][i] = matrix[j][i] / sum;
		}
	}
	node->setNormalizedMatrix(nMatrix);

	for(i=0; i< size; i++)
		free(nMatrix[i]);
	free(nMatrix);

	iterateFuncG(&AHPG::buildNormalizedmatrixG, node);
}

void AHPG::buildPmlG(Node* node) {
	int i,j;
	int size = node->getSize();
	if ( size == 0 ) return;
	float sum = 0;
	float* pml = (float*) malloc (sizeof(float) * size);
	float** matrix = node->getNormalizedMatrix();
	for (i = 0; i < size; i++) {
		sum = 0;
		for (j = 0; j < size; j++) {
			sum += matrix[i][j];
		}
		pml[i] = sum / (float)size;
	}
	node->setPml(pml);
	iterateFuncG(&AHPG::buildPmlG, node);
	free(pml);
}

void AHPG::buildPgG(Node* node) {
	int i;
	int size = this->hierarchy->getAlternativesSize();
	if ( size == 0 ) return;
	float* pg = (float*) malloc (sizeof(float) * size);
	for (i = 0; i < size; i++) {
		pg[i] = partialPgG(node, i);
	}
	node->setPg(pg, size);
	free(pg);
}

WeightType AHPG::partialPgG(Node* node, int alternative) {
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
	int i;
	int size = node->getSize();
	float** matrix = node->getMatrix();
	for (i = 0; i < size; i++)
		free(matrix[i]);
	free(matrix);
	matrix = NULL;
	node->setMatrix(NULL);
	iterateFuncG(&AHPG::deleteMatrixG, node);
}

void AHPG::deleteNormalizedMatrixG(Node* node) {
	int i;
	int size = node->getSize();
	float** nMatrix = node->getNormalizedMatrix();
	for (i = 0; i < size; i++)
		free(nMatrix[i]);
	free(nMatrix);
	nMatrix = NULL;
	node->setNormalizedMatrix(NULL);
	iterateFuncG(&AHPG::deleteNormalizedMatrixG, node);
}

void AHPG::deletePmlG(Node* node){
	int size = node->getSize();
	float* pml = node->getPml();
	free(pml);
	pml = NULL;
	node->setPml(NULL);
	iterateFuncG(&AHPG::deletePmlG, node);
}

void AHPG::checkConsistencyG(Node* node) {
	int i, j;
	int size = node->getSize();
	float** matrix = node->getMatrix();
	float* pml = node->getPml();
	float p[size], lambda = 0, RC = 0;
	for (i = 0; i < size; i++) {
		p[i] = 0;
		for (j = 0; j < size; j++) {
			p[i] += pml[j] * matrix[i][j];
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
	float** matrix = node->getMatrix();
	int tam = node->getSize();
	printf("Matrix of %s\n", node->getName());
	for (i = 0; i < tam; i++) {
		for (j = 0; j < tam; j++) {
			printf("%010lf\t", matrix[i][j]);
		}
		printf("\n");
	}
	printf("\n");
	iterateFuncG(&AHPG::printMatrixG, node);
}

void AHPG::printNormalizedMatrixG(Node* node) {
	int i,j;
	float **matrix = node->getNormalizedMatrix();
	int tam = node->getSize();
	printf("Normalized Matrix of %s\n", node->getName());
	for (i = 0; i < tam; i++) {
		for (j = 0; j < tam; j++) {
			printf("%010lf\t", matrix[i][j]);
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
	for (auto &hierarchyObject : dataObjective->value.GetObject()) {
		if (strcmp(hierarchyObject.name.GetString(), "name") == 0) {
			this->hierarchy->addFocus(strToLowerG(hierarchyObject.value.GetString())); // create the Focus* in the hierarchy;
		} else if (strcmp(hierarchyObject.name.GetString(), "childs") == 0) {
			criteriasParserG(&hierarchyObject, this->hierarchy->getFocus());
		} else {
			std::cout << "AHP -> Unrecognizable Type\nExiting...\n";
			exit(0);
		}
	}
}

void AHPG::criteriasParserG(genericValue* dataCriteria, Node* parent) {
	char* name;
	bool leaf = false;
	float* weight;
	int index=0;
	for (auto &childArray : dataCriteria->value.GetArray()) {
		weight = NULL;
		for (auto &child : childArray.GetObject()) {
			const char* n = child.name.GetString();
			if (strcmp(n, "name") == 0) {
				name = strToLowerG(child.value.GetString());
			} else if (strcmp(n, "leaf") == 0) {
				leaf = child.value.GetBool();
			} else if (strcmp(n, "weight") == 0) {
				for (auto &weightChild : child.value.GetArray()) {
					weight[index]=weightChild.GetFloat();
					index++;
				}
			} else if (strcmp(n, "childs") == 0) {
				// at this point, all the criteria variables were read, now the document
				// has the child's of the criteria. To put the childs corretly inside
				// the hierarchy, the criteria node has to be created.
				auto criteria = this->hierarchy->addCriteria(name);
				criteria->setLeaf(leaf);
				this->hierarchy->addEdge(parent, criteria, weight);
				// with the criteria node added, the call recursively the
				// criteriasParser.
				criteriasParserG(&child, criteria);
			}
		}
		if (leaf) {
			auto criteria = this->hierarchy->addCriteria(name);
			criteria->setLeaf(leaf);
			this->hierarchy->addSheets(criteria);
			this->hierarchy->addEdge(parent, criteria, weight);
		}
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
		rapidjson::SchemaDocument resourcesSchema =
			JSON::generateSchema("multicriteria/json/resourcesSchema.json");
		rapidjson::Document resourcesData =
			JSON::generateDocument("multicriteria/json/resourcesData.json");
		rapidjson::SchemaValidator resourcesValidator(resourcesSchema);
		if (!resourcesData.Accept(resourcesValidator))
			JSON::jsonError(&resourcesValidator);
		domParserG(&resourcesData);
		generateContentSchemaG();
	}
	// After reading the resoucesData, new alternativesSchema has to be created.
	// Parser the Json File that contains the Hierarchy
	rapidjson::SchemaDocument hierarchySchema =
		JSON::generateSchema("multicriteria/json/hierarchySchema.json");
	rapidjson::Document hierarchyData =
		JSON::generateDocument("multicriteria/json/hierarchyData.json");
	rapidjson::SchemaValidator hierarchyValidator(hierarchySchema);
	if (!hierarchyData.Accept(hierarchyValidator))
		JSON::jsonError(&hierarchyValidator);
	domParserG(&hierarchyData);
	if (alternativeParser) {
		// The Json Data is valid and can be used to construct the hierarchy.
		rapidjson::SchemaDocument alternativesSchema =
			JSON::generateSchema("multicriteria/json/alternativesSchema.json");
		rapidjson::Document alternativesData = JSON::generateDocument(
			"multicriteria/json/alternativesDataDefault.json");
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
	int i,j,k;
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
			for( j=0; j<sheetsSize; j++ ) {
				value = sheets[j]->getResource()->getResource(i);
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
				min_max_values[i] = (max-min)/ 9; // the other variables
			}
		}
	}

	// Create the data host memory
	int data_size = sheetsSize*altSize* altSize;
	int result_size = sheetsSize*altSize*altSize;

	float* h_data = (float*) malloc (sizeof(float)* data_size );

	{
		// TODO CAUNTION POSSIBLE ERROR
		int index=0;
		Edge** edges;
		float* temp_weights;
		for(i=0; i<sheetsSize; i++) {
			edges = sheets[i]->getEdges();
			for(j=0; j<altSize; j++) {
				temp_weights = edges[j]->getWeights();
				for(k = 0; k<edges[j]->getSize(); k++) {
					h_data[index] = temp_weights[k];
					index++;
				}
			}
		}
		edges = NULL;
		temp_weights = NULL;
	}

	// Allocate memory on the device
	dev_array<float> d_data(data_size);
	dev_array<float> d_min_max(resourceSize);
	dev_array<float> d_result(result_size);

	d_data.set(&h_data[0], data_size);
	d_min_max.set(&min_max_values[0], resourceSize);

	free(h_data);
	free(min_max_values);

	float* result = (float*) malloc (sizeof(float) * result_size);

	dim3 threadsPerBlock(block_size,block_size);
	dim3 numBlocks( ceil(altSize/(float)threadsPerBlock.x), ceil(altSize/(float)threadsPerBlock.y));


	acquisitonKernel<<< numBlocks, threadsPerBlock >>> (d_data.getData(), d_min_max.getData(), d_result.getData(), data_size, result_size);
	cudaDeviceSynchronize();

	d_result.get(&result[0], result_size);
	cudaDeviceSynchronize();

	// With all the weights calculated, now the weights are set in each edge between the sheets and alternatives
	Edge** edges = sheets[i]->getEdges(); // get the array of edges' pointer
	float temp[altSize];
	for(int i=0; i< sheetsSize; i++) {
		for (j = 0; j < altSize; j++) {         // iterate trhough all the edges
			memcpy(&temp, &result[i*sheetsSize+j*altSize], altSize);
			edges[j]->setWeights(temp, altSize);
		}
	}
}

void AHPG::synthesisG() {
	// 1 - Build the construccd the matrix
	buildMatrixG(this->hierarchy->getFocus());
	// printMatrix(this->hierarchy->getFocus());
	// 2 - Normalize the matrix
	buildNormalizedmatrixG(this->hierarchy->getFocus());
	// printNormalizedMatrix(this->hierarchy->getFocus());
	deleteMatrixG(this->hierarchy->getFocus());
	// 3 - calculate the PML
	buildPmlG(this->hierarchy->getFocus());
	deleteNormalizedMatrixG(this->hierarchy->getFocus());
	// printPml(this->hierarchy->getFocus());
	// 4 - calculate the PG
	buildPgG(this->hierarchy->getFocus());
	// printPg(this->hierarchy->getFocus());
	// Print all information
}

void AHPG::consistencyG() {
	iterateFuncG( &AHPG::checkConsistencyG, hierarchy->getFocus() );

}

// void AHPG::run(std::vector<Hierarchy<VariablesType,WeightType>::Alternative*>
// alt){
void AHPG::run(Host** alternatives, int size) {
	if (size == 0) {
		this->conceptionG(true);
	} else {
		// this->hierarchy->clearAlternatives(); // made in the setAlternatives function
		this->hierarchy->clearResource();

		Resource *resource = alternatives[0]->getResource();

		for (auto it : resource->mInt) {
			this->hierarchy->addResource((char*)it.first.c_str());
		}
		for (auto it : resource->mWeight) {
			this->hierarchy->addResource((char*)it.first.c_str());
		}
		for (auto it : resource->mBool) {
			this->hierarchy->addResource((char*)it.first.c_str());
		}
		this->conceptionG(false);
		this->setAlternatives(alternatives, size);
	}
	this->acquisitionG();
	this->synthesisG();
	// this->consistency();
}

std::map<int,char*> AHPG::getResult() {
	std::map<int,char*> result;
	float* values = this->hierarchy->getFocus()->getPg();
	std::vector<std::pair<int, float> > alternativesPair;

	unsigned int i;

	for (i = 0; i < this->hierarchy->getAlternativesSize(); i++) {
		alternativesPair.push_back(std::make_pair(i, values[i]));
	}
	// Nao e necessario fazer sort, o map ja realiza o sort do map pela chave em ordem acendente (menor - maior)
	// std::sort(alternativesPair.begin(), alternativesPair.end(),
	//           [](auto &left, auto &right) {
	//      return left.second > right.second;
	// });

	char* name;

	auto alternatives = this->hierarchy->getAlternatives();
	for (i = 0; i < (unsigned int)alternativesPair.size(); i++) {
		name = alternatives[alternativesPair[i].first]->getName();
		result[i+1] = name;
	}
	return result;
}

void AHPG::setAlternatives(Host** alternatives, int size) {
	int i;

	this->hierarchy->clearAlternatives();

	Resource* resource;

	for ( i=0; i<size; i++) {
		resource = alternatives[i]->getResource(); // Host resource

		Node* a = new Node(); // create the new node

		a->setResource(*this->hierarchy->getResource()); // set the default resources in the node

		a->setName((char*) alternatives[i]->getName().c_str()); // set the node name

		// Update the node h_resource values by the host resource values
		for (auto it : resource->mInt) {
			a->setResource((char*)it.first.c_str(), (float) it.second);
		}
		for (auto it : resource->mWeight) {
			a->setResource((char*)it.first.c_str(), it.second);
		}
		for (auto it : resource->mBool) {
			a->setResource((char*)it.first.c_str(), (float) it.second);
		}

		this->hierarchy->addAlternative(a);
	}

	this->hierarchy->addEdgeSheetsAlternatives();
}
