#include "ahp.hpp"

AHP::AHP() {
	this->hierarchy = NULL;
	IR[3] = 0.5245f;
	IR[4] = 0.8815f;
	IR[5] = 1.1086f;
	IR[6] = 1.1279f;
	IR[7] = 1.3417f;
	IR[8] = 1.4056f;
	IR[9] = 1.4499f;
	IR[10] = 1.4854f;
	IR[11] = 1.5141f;
	IR[12] = 1.5365f;
	IR[13] = 1.5551f;
	IR[14] = 1.5713f;
	IR[15] = 1.5838f;
	char cwd[1024];
	char* result;
	result = getcwd(cwd, sizeof(cwd));
	if(result == NULL) {
		SPDLOG_ERROR("The directory path is not correct");
	}
	char* sub_path = (strstr(cwd, "ahp"));
	if(sub_path!=NULL) {
		long int position = sub_path - cwd;
		strncpy(this->path, cwd, position);
		this->path[position-1] = '\0';
	}else{
		strcpy(this->path, cwd);
		strcat(this->path,"/");
	}
	this->setHierarchy();
	this->conception();
}

AHP::~AHP(){
	delete(hierarchy);
	IR.clear();
}

void AHP::setHierarchy(){
	if(this->hierarchy!=NULL)
		delete(this->hierarchy);

	this->hierarchy = new Hierarchy();
}

char* AHP::strToLower(const char* str) {
	unsigned int i;
	char* res = (char*) malloc( strlen(str)+1 );
	for(i=0; i < strlen(str); i++) {
		res[i] = static_cast<char>(tolower(str[i]));
	}
	res[strlen(str)] = '\0';
	return res;
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
	if ( size == 0 ) return;
	float* matrix = (float*) malloc (sizeof(float) * size*size);

	float* weights;

	for (i = 0; i < size; i++) {
		matrix[i*size+i] = 1;
		weights = (node->getEdges())[i]->getWeights();
		for (j = i + 1; j < size; j++) {
			if(weights == NULL ) {
				SPDLOG_ERROR("WEIGHT NULL");
				exit(0);
			}
			matrix[i*size+j] = weights[j];
			matrix[j*size+i] = 1 / matrix[i*size+j];
		}
	}

	node->setMatrix(matrix);
	iterateFunc(&AHP::buildMatrix, node);
}

void AHP::buildNormalizedmatrix(Node* node) {
	int i,j;
	int size = node->getSize();
	if ( size == 0 ) return;
	float* matrix = node->getMatrix(), sum = 0;
	float* nMatrix = (float*) malloc (sizeof(float) * size*size);

	for (i = 0; i < size; i++) {
		sum = 0;
		for (j = 0; j < size; j++) {
			sum += matrix[j*size+i];
		}
		for (j = 0; j < size; j++) {
			nMatrix[j*size+i] = matrix[j*size+i] / sum;
		}
	}
	node->setNormalizedMatrix(nMatrix);
	deleteMatrix(node);
	iterateFunc(&AHP::buildNormalizedmatrix, node);
}

void AHP::buildPml(Node* node) {
	int i,j;
	int size = node->getSize();
	if ( size == 0 ) return;
	float sum = 0;
	float* pml = (float*) malloc (sizeof(float) * size);
	float* matrix = node->getNormalizedMatrix();
	for (i = 0; i < size; i++) {
		sum = 0;
		for (j = 0; j < size; j++) {
			sum += matrix[i*size+j];
		}
		pml[i] = sum / (float)size;
	}
	node->setPml(pml);
	deleteNormalizedMatrix(node);
	iterateFunc(&AHP::buildPml, node);
}

void AHP::buildPg(Node* node) {
	int i;
	int size = this->hierarchy->getAlternativesSize();
	if ( size == 0 ) return;
	float* pg = (float*) malloc (sizeof(float) * size);
	for (i = 0; i < size; i++) {
		pg[i] = partialPg(node, i);
	}
	node->setPg(pg);
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
	float* matrix = node->getMatrix();
	free(matrix);
	matrix = NULL;
	node->setMatrix(NULL);
	// iterateFunc(&AHP::deleteMatrix, node);
}

void AHP::deleteNormalizedMatrix(Node* node) {
	float* nMatrix = node->getNormalizedMatrix();
	free(nMatrix);
	nMatrix = NULL;
	node->setNormalizedMatrix(NULL);
	// iterateFunc(&AHP::deleteNormalizedMatrix, node);
}

void AHP::deletePml(Node* node){
	float* pml = node->getPml();
	free(pml);
	pml = NULL;
	node->setPml(NULL);
	iterateFunc(&AHP::deletePml, node);
}

void AHP::checkConsistency(Node* node) {
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
	float f_size = static_cast<float>(size);
	lambda /= f_size;
	if (IR[size] > 0) {
		RC = static_cast<float>((fabs(lambda - f_size) / (f_size - 1)) / IR[size]);
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
		RC = static_cast<float>((abs(lambda - f_size) / (1.7699 * size - 4.3513)));
	}
	if (RC > 0.1) {
		SPDLOG_ERROR("Criteria {} is inconsistent", node->getName());
		spdlog::debug("RC= {}", RC);
		spdlog::debug("SIZE= {}", size);
		printMatrix(node);
		printNormalizedMatrix(node);
		printPml(node);
		exit(0);
	}
	iterateFunc(&AHP::checkConsistency, node);
}

void AHP::printMatrix(Node* node) {
	int i,j;
	float* matrix = node->getMatrix();
	int tam = node->getSize();
	if(tam==0) return;
	spdlog::debug("Matrix of {}", node->getName());
	for (i = 0; i < tam; i++) {
		for (j = 0; j < tam; j++) {
			spdlog::debug("{}\t", matrix[i*tam+j]);
		}
	}
	spdlog::debug("");
	iterateFunc(&AHP::printMatrix, node);
}

void AHP::printNormalizedMatrix(Node* node) {
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
	iterateFunc(&AHP::printNormalizedMatrix, node);
}

void AHP::printPml(Node* node) {
	int i;
	float* pml = node->getPml();
	int tam = node->getSize();
	if(tam==0) return;
	spdlog::debug("PML of {}", node->getName());
	for (i = 0; i < tam; i++) {
		spdlog::debug("{}\t", pml[i]);
	}
	spdlog::debug("");
	iterateFunc(&AHP::printPml, node);
}

void AHP::printPg(Node* node) {
	int i;
	float* pg = node->getPg();
	int tam = this->hierarchy->getAlternativesSize();
	if(tam==0) return;
	spdlog::debug("PG of {}", node->getName());
	for (i = 0; i < tam; i++) {
		spdlog::debug("{}\t",pg[i]);
	}
	spdlog::debug("");
}

void AHP::hierarchyParser(const rapidjson::Value &hierarchyData) {
	Node* focus= this->hierarchy->addFocus(hierarchyData["name"].GetString());
	Node* criteria = NULL;

	const rapidjson::Value &c_array = hierarchyData["childs"];
	unsigned int c_size = c_array.Size();
	float weights[c_size];

	for(unsigned int i=0; i< c_size; i++) {
		criteria = this->hierarchy->addCriteria(c_array[i]["name"].GetString());
		const rapidjson::Value & w_array = c_array[i]["weight"];
		for(unsigned int j=0; j<w_array.Size(); j++) {
			weights[j] = w_array[j].GetFloat();
		}
		this->hierarchy->addEdge(focus, criteria, weights, c_size);
	}
}

void AHP::conception() {
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
	else if(this->type==2)
		strcat(hierarchy_data, "multicriteria/ahp/json/hierarchyDataBW.json");
	else
		SPDLOG_ERROR("Hierarchy data type error");

	rapidjson::SchemaDocument hierarchySchema =
		JSON::generateSchema(hierarchy_schema);
	rapidjson::Document hierarchyData =
		JSON::generateDocument(hierarchy_data);
	rapidjson::SchemaValidator hierarchyValidator(hierarchySchema);

	if (!hierarchyData.Accept(hierarchyValidator))
		JSON::jsonError(&hierarchyValidator);
	hierarchyParser(hierarchyData["objective"]);
}

void AHP::acquisition() {
	int i,j,k;
	// Para gerar os pesos das alterntivas, será primeiro captado o MIN e MAX
	// valor das alternativas , após isso será montada as matrizes de cada sheet
	// auto max = this->hierarchy->getResource();
	// auto min = this->hierarchy->getResource();
	Node** alt = this->hierarchy->getAlternatives();
	Node** sheets = this->hierarchy->getCriterias();

	const int altSize = this->hierarchy->getAlternativesSize();
	const int sheetsSize = this->hierarchy->getCriteriasSize();
	const int resourceSize = this->hierarchy->getResource()->getDataSize();

	spdlog::debug("\tbuilding the min max values");
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
				min_max_values[i] = static_cast<float>((max-min)/ 9.0); // the other variables
			}
		}
	}
	// At this point, all the integers and float/float resources  has
	// the max and min values discovered.
	spdlog::debug("\tbuilding the criteria weights");
	float** criteriasWeight = (float**) malloc (sizeof(float*) * altSize);
	for(i=0; i<altSize; i++) {
		criteriasWeight[i] = (float*) malloc (sizeof(float*) * altSize);
	}

	spdlog::debug("\tcalculating the results for {} sheets", sheetsSize);
	float result;
	for ( i=0; i< sheetsSize; i++) {
		for ( j=0; j<altSize; j++) {
			for ( k=0; k<altSize; k++) {
				if(min_max_values[i]==0) {
					result=0;
				}
				else if(min_max_values[i]!=-1) {
					result = (alt[j]->getResource()->getResource(i) - alt[k]->getResource()->getResource(i)) /min_max_values[i];
				}else{         // boolean resource
					if(alt[j]->getResource()->getResource(i) == alt[k]->getResource()->getResource(i)) {
						result = 1;
					}else{
						alt[j]->getResource()->getResource(i) ? result = 9 : result = static_cast<float>(1 / 9.0);
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
		for (j = 0; j < altSize; j++) { // iterate trhough all the edges
			edges[j]->setWeights(criteriasWeight[j], altSize);
		}
	}

	spdlog::debug("\tfree the allocated temp data");
	for(i=0; i<altSize; i++) {
		free(criteriasWeight[i]);
	}

	free(criteriasWeight);
	free(min_max_values);
}

void AHP::synthesis() {
	// 1 - Build the construccd the matrix
	// printf("B M\n");
	buildMatrix(this->hierarchy->getFocus());
	// printMatrix(this->hierarchy->getFocus());
	// 2 - Normalize the matrix
	// printf("B N\n");
	buildNormalizedmatrix(this->hierarchy->getFocus());
	// printNormalizedMatrix(this->hierarchy->getFocus());
	// printf("D M\n");
	// 3 - calculate the PML
	// printf("B P\n");
	buildPml(this->hierarchy->getFocus());
	// printPml(this->hierarchy->getFocus());
	// 4 - calculate the PG
	// printf("B PG\n");
	buildPg(this->hierarchy->getFocus());
	// printPg(this->hierarchy->getFocus());
	// Print all information
}

void AHP::consistency() {
	iterateFunc( &AHP::checkConsistency, hierarchy->getFocus() );
}

void AHP::run(Host** alternatives, int size, int interval_low, int interval_high) {
	spdlog::debug("AHP in [{},{}]", interval_low, interval_high);
	this->hierarchy->clearAlternatives();
	this->hierarchy->clearResource();

	//As the resource is the interval tree, it is considerable to think that each interval can be view as its area, multiplying the capacity that it consumes by the time it demands.
	// float consimed = (high - low) * capacity (a area in 2D view)
	std::map<std::string, Interval_Tree::Interval_Tree*> resource = alternatives[0]->getResource();

	spdlog::debug("Adding resources into the hierarchy");
	for (auto it : resource) {
		this->hierarchy->addResource((char*)it.first.c_str());
	}

	spdlog::debug("Setting the alternatives");
	this->setAlternatives(alternatives, size, interval_low, interval_high);
	if(this->hierarchy->getCriteriasSize()==0) {
		SPDLOG_ERROR("AHP Hierarchy with no sheets");
		exit(0);
	}

	spdlog::debug("Running acquisition");
	this->acquisition();
	spdlog::debug("Running synthesis");
	this->synthesis();
	spdlog::debug("AHP[x]");
}

unsigned int* AHP::getResult(unsigned int& size) {

	unsigned int* result = (unsigned int*) malloc(sizeof(unsigned int));
	float* values = this->hierarchy->getFocus()->getPg();
	std::priority_queue<std::pair<float, unsigned int> > alternativesPair;

	unsigned int i;

	// printf("PG\n");
	for (i = 0; i < (unsigned int) this->hierarchy->getAlternativesSize(); i++) {
		// if( (unsigned int)alternativesPair.size()<20)
		// printf("%f#%u$ ",values[i],i);
		alternativesPair.push(std::make_pair(values[i], i));
	}

	Node** alternatives = this->hierarchy->getAlternatives();

	i=0;
	while(!alternativesPair.empty()) {
		//for (i = 0; i < (unsigned int)alternativesPair.size(); i++) {
		result = (unsigned int*) realloc (result, sizeof(unsigned int)*(i+1));
		result[i] =  atoi(alternatives[alternativesPair.top().second]->getName());
		alternativesPair.pop();
		i++;
	}
	size = i;
	return result;
}

void AHP::setAlternatives(Host** alternatives, int size, int low, int high) {
	std::map<std::string, Interval_Tree::Interval_Tree*> resource;
	Node* alt = NULL;
	float temp_capacity = 0;
	for ( size_t i=0; i < (size_t)size; i++) {
		resource = alternatives[i]->getResource(); // Host resource

		alt = new Node(); // create the new node

		alt->setResource(this->hierarchy->getResource()); // set the default resources in the node

		alt->setName((char*) std::to_string(alternatives[i]->getId()).c_str()); // set the node name

		// Update the node h_resource values by the host resource values
		spdlog::debug("Looping resources");
		for (auto it : resource) {
			spdlog::debug("Configuring the temp capacity for {} in [{},{}] in the host {}", it.first, low, high, alternatives[i]->getId());
			it.second->show();
			temp_capacity = static_cast<float>((high - low)) * it.second->getMinValueAvailable(low, high);
			spdlog::debug("Setting alt temp_capa: {}", temp_capacity);
			alt->setResource((char*)it.first.c_str(), temp_capacity);
			spdlog::debug("Alt set");

		}
		spdlog::debug("Adding alternative");
		this->hierarchy->addAlternative(alt);
		spdlog::debug("Added");
	}

	this->hierarchy->addEdgeCriteriasAlternatives();
}

void AHP::readJson(){
}
