#include "ahp.hpp"

AHP::AHP() {
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
	getcwd(cwd, sizeof(cwd));

	char* sub_path = (strstr(cwd, "multicriteria"));
	int position = sub_path - cwd;
	strncpy(this->path, cwd, position);
	this->path[position] = '\0';
}

AHP::~AHP(){
	delete(hierarchy);
}

void AHP::setHierarchy(){
	this->hierarchy = new Hierarchy();
}

char* AHP::strToLower(const char* str) {
	int i;
	char* res = (char*) malloc( strlen(str)+1 );
	for(i=0; i<strlen(str); i++) {
		res[i]=tolower(str[i]);
	}
	res[strlen(str)] = '\0';
	return res;
}

void AHP::updateAlternatives() {
	char cwd[1024];
	getcwd(cwd, sizeof(cwd));
	char* path = strtok(cwd, "multicriteria");
	printf("PATH %s\n",path);

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
template <typename F> void AHP::iterateFunc(F f, Node* node) {
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

void AHP::buildMatrix(Node* node) {
	int i,j;
	int size = node->getSize(); // get the number of edges
	printf("SIZE %d\n");
	if ( size == 0 ) return;
	float** matrix = (float**) malloc (sizeof(float*) * size);
	for (i = 0; i < size; i++)
		matrix[i] = (float*) malloc (sizeof(float) * size);

	float* weights;

	for (i = 0; i < size; i++) {
		printf("I = %d\n",i);
		matrix[i][i] = 1;
		weights = (node->getEdges())[i]->getWeights();
		for (j = i + 1; j < size; j++) {
			if(weights == NULL ) {
				printf("ERRO BRUSCO BUILD MATRIX WEIGHT = NULL\n");
				exit(0);
			}
			printf("J = %d\n",j);
			printf("Matrix[%d][%d]\n",i,j);
			printf("Weights[%d] = %f",j,weights[j]);
			matrix[i][j] = weights[j];
			matrix[j][i] = 1 / matrix[i][j];
		}
	}

	node->setMatrix(matrix);

	for(i=0; i< size; i++)
		free(matrix[i]);
	free(matrix);

	iterateFunc(&AHP::buildMatrix, node);
}

void AHP::buildNormalizedmatrix(Node* node) {
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

	iterateFunc(&AHP::buildNormalizedmatrix, node);
}

void AHP::buildPml(Node* node) {
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
	iterateFunc(&AHP::buildPml, node);
	free(pml);
}

void AHP::buildPg(Node* node) {
	int i;
	int size = this->hierarchy->getAlternativesSize();
	if ( size == 0 ) return;
	float* pg = (float*) malloc (sizeof(float) * size);
	for (i = 0; i < size; i++) {
		pg[i] = partialPg(node, i);
	}
	node->setPg(pg, size);
	free(pg);
}

float AHP::partialPg(Node* node, int alternative) {
	int i;

	Node* criteria;

	Edge** edges= node->getEdges();
	int size = node->getSize();

	float* pml = node->getPml();
	float partial = 0;
	for (i = 0; i < size; i++) {
		criteria = edges[i]->getNode();
		if (criteria != NULL && criteria->getType()!=node_t::ALTERNATIVE) {
			partial += pml[i] * partialPg(criteria, alternative);
		} else {
			return pml[alternative];
		}
	}
	return partial;
}

void AHP::deleteMatrix(Node* node) {
	int i;
	int size = node->getSize();
	float** matrix = node->getMatrix();
	for (i = 0; i < size; i++)
		free(matrix[i]);
	free(matrix);
	matrix = NULL;
	node->setMatrix(NULL);
	iterateFunc(&AHP::deleteMatrix, node);
}

void AHP::deleteNormalizedMatrix(Node* node) {
	int i;
	int size = node->getSize();
	float** nMatrix = node->getNormalizedMatrix();
	for (i = 0; i < size; i++)
		free(nMatrix[i]);
	free(nMatrix);
	nMatrix = NULL;
	node->setNormalizedMatrix(NULL);
	iterateFunc(&AHP::deleteNormalizedMatrix, node);
}

void AHP::deletePml(Node* node){
	int size = node->getSize();
	float* pml = node->getPml();
	free(pml);
	pml = NULL;
	node->setPml(NULL);
	iterateFunc(&AHP::deletePml, node);
}

void AHP::checkConsistency(Node* node) {
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
		printMatrix(node);
		printNormalizedMatrix(node);
		printPml(node);
		exit(0);
	}
	iterateFunc(&AHP::checkConsistency, node);
}

void AHP::printMatrix(Node* node) {
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
	iterateFunc(&AHP::printMatrix, node);
}

void AHP::printNormalizedMatrix(Node* node) {
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
	iterateFunc(&AHP::printNormalizedMatrix, node);
}

void AHP::printPml(Node* node) {
	int i;
	float* pml = node->getPml();
	int tam = node->getSize();
	printf("PML of %s\n", node->getName());
	for (i = 0; i < tam; i++) {
		printf("%010lf\t", pml[i]);
	}
	printf("\n");
	iterateFunc(&AHP::printPml, node);
}

void AHP::printPg(Node* node) {
	int i;
	float* pg = node->getPg();
	int tam = this->hierarchy->getAlternativesSize();
	printf("PG of %s\n", node->getName());
	for (i = 0; i < tam; i++) {
		printf("%010lf\t",pg[i]);
	}
	printf("\n");
}

void AHP::generateContentSchema() {
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

void AHP::resourcesParser(genericValue* dataResource) {
	std::string variableName, variableType;
	for (auto &arrayData : dataResource->value.GetArray()) {
		variableName = variableType = "";
		for (auto &objectData : arrayData.GetObject()) {
			if (strcmp(objectData.name.GetString(), "name") == 0) {
				variableName = objectData.value.GetString();
			} else if (strcmp(objectData.name.GetString(), "variableType") == 0) {
				variableType = strToLower(objectData.value.GetString());
			} else {
				std::cout << "Error in reading resources\nExiting...\n";
				exit(0);
			}
		}
		this->hierarchy->addResource((char*)variableName.c_str());
	}
}

void AHP::hierarchyParser(genericValue* dataObjective) {
	for (auto &hierarchyObject : dataObjective->value.GetObject()) {
		if (strcmp(hierarchyObject.name.GetString(), "name") == 0) {
			this->hierarchy->addFocus(strToLower(hierarchyObject.value.GetString())); // create the Focus* in the hierarchy;
		} else if (strcmp(hierarchyObject.name.GetString(), "childs") == 0) {
			criteriasParser(&hierarchyObject, this->hierarchy->getFocus() );
		} else {
			std::cout << "AHP -> Unrecognizable Type\nExiting...\n";
			exit(0);
		}
	}
}

void AHP::criteriasParser(genericValue* dataCriteria, Node* parent) {
	char* name;
	bool leaf = false;
	float* weight = NULL;
	int index=0;
	for (auto &childArray : dataCriteria->value.GetArray()) {
		weight = NULL;
		for (auto &child : childArray.GetObject()) {
			const char* n = child.name.GetString();
			if (strcmp(n, "name") == 0) {
				name = strToLower(child.value.GetString());
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
				this->hierarchy->addEdge(parent, criteria, weight);
				// with the criteria node added, the call recursively the
				// criteriasParser.
				criteriasParser(&child, criteria);
			}
		}
		if (leaf) {
			printf("Setting a leaf\n");
			auto criteria = this->hierarchy->addCriteria(name);
			criteria->setLeaf(leaf);
			this->hierarchy->addSheets(criteria);
			this->hierarchy->addEdge(parent, criteria, weight);
		}
		free(weight);
	}
}

void AHP::alternativesParser(genericValue* dataAlternative) {
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
					alternative->setName(strToLower(alt.value.GetString()));
				}
			}
		}
	}
}

void AHP::domParser(rapidjson::Document *data) {
	for (auto &m : data->GetObject()) { // query through all objects in data.
		if (strcmp(m.name.GetString(), "resources") == 0) {
			resourcesParser(&m);
		} else if (strcmp(m.name.GetString(), "objective") == 0) {
			hierarchyParser(&m);
		} else if (strcmp(m.name.GetString(), "alternatives") == 0) {
			alternativesParser(&m);
		}
	}
}

void AHP::conception(bool alternativeParser) {
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
		domParser(&resourcesData);
		generateContentSchema();
	}
	// After reading the resoucesData, new alternativesSchema has to be created.
	// Parser the Json File that contains the Hierarchy
	char hierarchy_schema [1024] = "\0";
	char hierarchy_data [1024] = "\0";

	strcpy(hierarchy_schema, path);
	strcpy(hierarchy_data, path);

	strcat(hierarchy_schema, "multicriteria/json/hierarchySchema.json");
	strcat(hierarchy_data, "multicriteria/json/hierarchyData.json");

	rapidjson::SchemaDocument hierarchySchema =
		JSON::generateSchema(hierarchy_schema);
	rapidjson::Document hierarchyData =
		JSON::generateDocument(hierarchy_data);
	rapidjson::SchemaValidator hierarchyValidator(hierarchySchema);

	if (!hierarchyData.Accept(hierarchyValidator))
		JSON::jsonError(&hierarchyValidator);
	domParser(&hierarchyData);

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
		domParser(&alternativesData);
		this->hierarchy->addEdgeSheetsAlternatives();
	}
}

void AHP::acquisition() {
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

	// At this point, all the integers and float/float resources  has
	// the max and min values discovered.
	float** criteriasWeight = (float**) malloc (sizeof(float*) * altSize);
	for(i=0; i<altSize; i++) {
		criteriasWeight[i] = (float*) malloc (sizeof(float*) * altSize);
	}

	float result;
	printf("SS %d AS %d\n", sheetsSize, altSize);
	for ( i=0; i< sheetsSize; i++) {
		for ( j=0; j<altSize; j++) {
			for ( k=0; k<altSize; k++) {

				if(min_max_values[i]!=-1) {
					result = (alt[j]->getResource()->getResource(i) - alt[k]->getResource()->getResource(i)) / min_max_values[i];
				}else{ // boolean resource
					if(alt[j]->getResource()->getResource(i) == alt[k]->getResource()->getResource(i)) {
						result = 1;
					}else{
						alt[j]->getResource()->getResource(i) ? result = 9 : result = 1 / 9.0;
					}
				}
				if (result == 0) {
					result = 1;
				} else if (result < 0) {
					result = (-1) / result;
				}
				criteriasWeight[j][k]=result;
			}
		}
		// With all the weights calculated, now the weights are set in each edge
		// between the sheets and alternatives
		Edge** edges = sheets[i]->getEdges(); // get the array of edges' pointer
		sheets[i]->setSize(altSize);
		printf("SHEETS SIZE = %d\n", this->hierarchy->getSheetsSize());
		for (j = 0; j < altSize; j++) { // iterate trhough all the edges
			edges[j]->setWeights(criteriasWeight[j], altSize);
		}
	}

	for(i=0; i<altSize; i++) {
		free(criteriasWeight[i]);
	}

	free(criteriasWeight);
	free(min_max_values);
}

void AHP::synthesis() {
	// 1 - Build the construccd the matrix
	printf("B M\n");
	buildMatrix(this->hierarchy->getFocus());
	// printMatrix(this->hierarchy->getFocus());
	// 2 - Normalize the matrix
	printf("B N\n");
	buildNormalizedmatrix(this->hierarchy->getFocus());
	// printNormalizedMatrix(this->hierarchy->getFocus());
	printf("D M\n");
	deleteMatrix(this->hierarchy->getFocus());
	// 3 - calculate the PML
	printf("B P\n");
	buildPml(this->hierarchy->getFocus());
	// printPml(this->hierarchy->getFocus());
	printf("D Nn");
	deleteNormalizedMatrix(this->hierarchy->getFocus());
	// 4 - calculate the PG
	printf("B PG\n");
	buildPg(this->hierarchy->getFocus());
	// printf("D P\n");
	// deletePml(this->hierarchy->getFocus());
	// printPg(this->hierarchy->getFocus());
	// Print all information
}

void AHP::consistency() {
	iterateFunc( &AHP::checkConsistency, hierarchy->getFocus() );
}

void AHP::run(Host** alternatives, int size) {
	if (size == 0) {
		this->conception(true);
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
		this->conception(false);
		this->setAlternatives(alternatives, size);
	}
	printf("OLa\n");
	this->acquisition();
	printf("Acquisiton done\n");
	this->synthesis();
	printf("Synthesis done\n");
	// this->consistency();
}

std::map<int,char*> AHP::getResult() {
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

void AHP::setAlternatives(Host** alternatives, int size) {
	int i;

	this->hierarchy->clearAlternatives();

	Resource* resource = NULL;
	Node* a = NULL;
	for ( i=0; i<size; i++) {
		resource = alternatives[i]->getResource(); // Host resource

		a = new Node(); // create the new node

		a->setResource(this->hierarchy->getResource()); // set the default resources in the node

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
