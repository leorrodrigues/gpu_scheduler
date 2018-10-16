#include "ahpg.cuh"

AHPG::AHPG() {
	this->hierarchy = new Hierarchy<VariablesType, WeightType>();
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

std::string AHPG::strToLowerG(std::string s) {
	std::transform(s.begin(), s.end(), s.begin(),
	               [](unsigned char c) {
		return std::tolower(c);
	});
	return s;
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
	domParserG(&alternativesData);
	this->hierarchy->addEdgeSheetsAlternatives();
}

// Recieve a function address to iterate, used in build matrix, normalized, pml
// and pg functions.
template <typename F, typename T> void AHPG::iterateFuncG(F f, T *v) {
	std::vector<Hierarchy<VariablesType, WeightType>::Edge *> e = v->getEdges();
	Hierarchy<VariablesType, WeightType>::Criteria *c;
	for (edgeIt it = e.begin(); it != e.end(); it++) {
		c = (*it)->getCriteria();
		if (c != NULL) {
			(this->*f)(c);
		}
	}
}

template <typename T> void AHPG::buildMatrixG(T *v) {
	int size = v->edgesCount();
	WeightType **matrix = new (std::nothrow) WeightType *[size];
	for (int i = 0; i < size; i++)
		matrix[i] = new (std::nothrow) WeightType[size];
	std::vector<WeightType> w;
	for (int i = 0; i < size; i++) {
		matrix[i][i] = 1;
		w = (v->getEdges())[i]->getWeights();
		for (int j = i + 1; j < size; j++) {
			matrix[i][j] = w[j];
			matrix[j][i] = 1 / matrix[i][j];
		}
	}
	v->setMatrix(matrix);
	iterateFuncG(
		&AHPG::buildMatrixG<Hierarchy<VariablesType, WeightType>::Criteria>, v);
}

template <typename T> void AHPG::buildNormalizedmatrixG(T *v) {
	int size = v->edgesCount();
	WeightType **matrix = v->getMatrix(), sum = 0;
	WeightType **nMatrix = new (std::nothrow) WeightType *[size];
	for (int i = 0; i < size; i++)
		nMatrix[i] = new (std::nothrow) WeightType[size];
	for (int i = 0; i < size; i++) {
		sum = 0;
		for (int j = 0; j < size; j++) {
			sum += matrix[j][i];
		}
		for (int j = 0; j < size; j++) {
			nMatrix[j][i] = matrix[j][i] / sum;
		}
	}
	v->setNormalizedMatrix(nMatrix);
	iterateFuncG(&AHPG::buildNormalizedmatrixG<
			     Hierarchy<VariablesType, WeightType>::Criteria>,
	             v);
}

template <typename T> void AHPG::buildPmlG(T *v) {
	int size = v->edgesCount();
	WeightType sum = 0;
	WeightType *pml = new (std::nothrow) WeightType[size];
	WeightType **matrix = v->getNormalizedMatrix();
	for (int i = 0; i < size; i++) {
		sum = 0;
		for (int j = 0; j < size; j++) {
			sum += matrix[i][j];
		}
		pml[i] = sum / size;
	}
	v->setPml(pml);
	iterateFuncG(&AHPG::buildPmlG<Hierarchy<VariablesType, WeightType>::Criteria>,
	             v);
}

template <typename T> void AHPG::buildPgG(T *v) {
	int aSize = this->hierarchy->getAlternativesCount();
	std::vector<Hierarchy<VariablesType, WeightType>::Edge *> e = v->getEdges();
	WeightType *pg = new (std::nothrow) WeightType[aSize];
	for (int i = 0; i < aSize; i++) {
		pg[i] = partialPgG(v, i);
	}
	v->setPg(pg);
}

template <typename T> WeightType AHPG::partialPgG(T *v, int alternative) {
	std::vector<Hierarchy<VariablesType, WeightType>::Edge *> e = v->getEdges();
	int size = e.size();
	Hierarchy<VariablesType, WeightType>::Criteria *c;
	WeightType *pml = v->getPml();
	WeightType partial = 0;
	for (int i = 0; i < size; i++) {
		c = e[i]->getCriteria();
		if (c != NULL) {
			partial += pml[i] * partialPgG(c, alternative);
		} else {
			return pml[alternative];
		}
	}
	return partial;
}

template <typename T> void AHPG::deleteMatrixG(T *v) {
	int size = v->edgesCount();
	WeightType **matrix = v->getMatrix();
	for (int i = 0; i < size; i++)
		delete[] matrix[i];
	delete[] matrix;
	matrix = NULL;
	v->setMatrix(NULL);
	iterateFuncG(
		&AHPG::deleteMatrixG<Hierarchy<VariablesType, WeightType>::Criteria>, v);
}

template <typename T> void AHPG::deleteNormalizedMatrixG(T *v) {
	int size = v->edgesCount();
	WeightType **nMatrix = v->getNormalizedMatrix();
	for (int i = 0; i < size; i++)
		delete[] nMatrix[i];
	delete[] nMatrix;
	nMatrix = NULL;
	v->setNormalizedMatrix(NULL);
	iterateFuncG(&AHPG::deleteNormalizedMatrixG<
			     Hierarchy<VariablesType, WeightType>::Criteria>,
	             v);
}

template <typename T> void AHPG::checkConsistencyG(T *v) {
	int size = v->edgesCount();
	WeightType **matrix = v->getMatrix();
	WeightType *pml = v->getPml();
	WeightType p[size], lambda = 0, RC = 0;
	for (int i = 0; i < size; i++) {
		p[i] = 0;
		for (int j = 0; j < size; j++) {
			p[i] += pml[j] * matrix[i][j];
		}
		lambda += (p[i] / pml[i]);
	}
	lambda /= size;
	if (IR[size] > 0) {
		RC = (abs(lambda - size) / (size - 1)) / IR[size];
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
		RC = (fabs(lambda - size) / (1.7699 * size - 4.3513));
	}
	if (RC > 0.1) {
		std::cout << "ERROR: Criteria: " << v->getName() << " is inconsistent\n";
		std::cout << "RC= " << RC << "\n";
		std::cout<<"SIZE: "<<size<<"\n";
		printMatrixG(v);
		printNormalizedMatrixG(v);
		printPmlG(v);
		exit(0);
	}
	iterateFuncG(
		&AHPG::checkConsistencyG<Hierarchy<VariablesType, WeightType>::Criteria>,
		v);
}

void AHPG::generateContentSchemaG() {
	std::string names;
	std::string text = "{\"$schema\":\"http://json-schema.org/draft-04/"
	                   "schema#\",\"definitions\": {\"alternative\": {\"type\": "
	                   "\"array\",\"minItems\": 1,\"items\":{\"properties\": {";
	auto resource = this->hierarchy->getResource();
	for (auto it : resource->mInt) {
		text += "\"" + it.first + "\":{\"type\":\"number\"},";
		names += "\"" + it.first + "\",";
	}
	for (auto it : resource->mWeight) {
		text += "\"" + it.first + "\":{\"type\":\"number\"},";
		names += "\"" + it.first + "\",";
	}
	for (auto it : resource->mBool) {
		text += "\"" + it.first + "\":{\"type\":\"boolean\"},";
		names += "\"" + it.first + "\",";
	}
	for (auto it : resource->mString) {
		text += "\"" + it.first + "\":{\"type\":\"string\"},";
		names += "\"" + it.first + "\",";
	}
	names.pop_back();
	text.pop_back();
	text += "},\"additionalProperties\": false,\"required\": [" + names +
	        "]}}},\"type\": \"object\",\"minProperties\": "
	        "1,\"additionalProperties\": false,\"properties\": "
	        "{\"alternatives\": {\"$ref\": \"#/definitions/alternative\"}}}";
	JSON::writeJson("multicriteria/json/alternativesSchema.json", text);
}

template <typename T> void AHPG::printMatrixG(T *v) {
	WeightType **matrix = v->getMatrix();
	int tam = v->edgesCount();
	std::cout << "Matrix of " << v->getName() << "\n";
	for (int i = 0; i < tam; i++) {
		for (int j = 0; j < tam; j++) {
			std::cout << std::setfill(' ') << std::setw(10) << matrix[i][j] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
	iterateFuncG(
		&AHPG::printMatrixG<Hierarchy<VariablesType, WeightType>::Criteria>, v);
}

template <typename T> void AHPG::printNormalizedMatrixG(T *v) {
	WeightType **matrix = v->getNormalizedMatrix();
	int tam = v->edgesCount();
	std::cout << "Normalized Matrix of " << v->getName() << "\n";
	for (int i = 0; i < tam; i++) {
		for (int j = 0; j < tam; j++) {
			std::cout << std::setfill(' ') << std::setw(10) << matrix[i][j] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
	iterateFuncG(&AHPG::printNormalizedMatrixG<
			     Hierarchy<VariablesType, WeightType>::Criteria>,
	             v);
}

template <typename T> void AHPG::printPmlG(T *v) {
	WeightType *pml = v->getPml();
	int tam = v->edgesCount();
	std::cout << "PML of " << v->getName() << "\n";
	for (int i = 0; i < tam; i++) {
		std::cout << std::setfill(' ') << std::setw(10) << pml[i] << " ";
	}
	std::cout << "\n";
	iterateFuncG(&AHPG::printPmlG<Hierarchy<VariablesType, WeightType>::Criteria>,
	             v);
}

template <typename T> void AHPG::printPgG(T *v) {
	WeightType *pg = v->getPg();
	int tam = this->hierarchy->getAlternativesCount();
	std::cout << "PG of " << v->getName() << "\n";
	for (int i = 0; i < tam; i++) {
		std::cout << std::setfill(' ') << std::setw(10) << pg[i] << " ";
	}
	std::cout << "\n";
}

void AHPG::resourcesParserG(genericValue *dataResource) {
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
		this->hierarchy->addResource(variableName, variableType);
	}
}

void AHPG::hierarchyParserG(genericValue *dataObjective) {
	for (auto &hierarchyObject : dataObjective->value.GetObject()) {
		if (strcmp(hierarchyObject.name.GetString(), "name") == 0) {
			this->hierarchy->addFocus(
				strToLowerG(hierarchyObject.value
				            .GetString())); // create the Focus* in the hierarchy;
		} else if (strcmp(hierarchyObject.name.GetString(), "childs") == 0) {
			criteriasParserG(&hierarchyObject, this->hierarchy->getFocus());
		} else {
			std::cout << "AHPG -> Unrecognizable Type\nExiting...\n";
			exit(0);
		}
	}
}

template <typename Parent>
void AHPG::criteriasParserG(genericValue *dataCriteria, Parent p) {
	std::string name = " ";
	bool leaf = false;
	std::vector<WeightType> weight;
	for (auto &childArray : dataCriteria->value.GetArray()) {
		weight.clear();
		for (auto &child : childArray.GetObject()) {
			const char *n = child.name.GetString();
			if (strcmp(n, "name") == 0) {
				name = strToLowerG(child.value.GetString());
			} else if (strcmp(n, "leaf") == 0) {
				leaf = child.value.GetBool();
			} else if (strcmp(n, "weight") == 0) {
				for (auto &weightChild : child.value.GetArray()) {
					weight.push_back(weightChild.GetDouble());
				}
			} else if (strcmp(n, "childs") == 0) {
				// at this point, all the criteria variables were read, now the document
				// has the child's of the criteria. To put the childs corretly inside
				// the hierarchy, the criteria node has to be created.
				auto criteria = this->hierarchy->addCriteria(name);
				criteria->setLeaf(leaf);
				this->hierarchy->addEdge(p, criteria, weight);
				// with the criteria node added, the call recursively the
				// criteriasParser.
				criteriasParserG(&child, criteria);
			}
		}
		if (leaf) {
			auto criteria = this->hierarchy->addCriteria(name);
			criteria->setLeaf(leaf);
			this->hierarchy->addSheets(criteria);
			this->hierarchy->addEdge(p, criteria, weight);
		}
	}
}

void AHPG::alternativesParserG(genericValue *dataAlternative) {
	for (auto &arrayAlternative : dataAlternative->value.GetArray()) {
		auto alternative = this->hierarchy->addAlternative();
		for (auto &alt : arrayAlternative.GetObject()) {
			std::string name(alt.name.GetString());
			if (alt.value.IsNumber()) {
				if (alternative->getResource()->mInt.count(name) > 0) {
					alternative->setResource(name, alt.value.GetInt());
				} else {
					alternative->setResource(name, alt.value.GetFloat());
				}
			} else if (alt.value.IsBool()) {
				alternative->setResource(name, alt.value.GetBool());
			} else {
				alternative->setResource(
					name, strToLowerG(std::string(alt.value.GetString())));
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

__device__
int char_to_int(char* str){
	int res = 0; // Initialize result
	int sign = 1;  // Initialize sign as positive
	int i = 0;  // Initialize index of first digit
	// If number is negative, then update sign
	if (str[0] == '-') {
		sign = -1;
		i++; // Also update index of first digit
	}
	// Iterate through all digits and update the result
	for (; str[i] != '\0'; ++i)
		res = res*10 + str[i] - '0';
	// Return result with sign
	return sign*res;
}

__device__
float char_to_float(char* str){
	float rez = 0, fact = 1;
	if (*str == '-') {
		str++;
		fact = -1;
	};
	for (int point_seen = 0; *str; str++) {
		if (*str == '.') {
			point_seen = 1;
			continue;
		};
		int d = *str - '0';
		if (d >= 0 && d <= 9) {
			if (point_seen) fact /= 10.0f;
			rez = rez * 10.0f + (float)d;
		};
	}
	// printf("%f AND %f\n",rez,fact);
	return rez*fact;
}

__device__
char* copyStr(char* data, int init, int end){
	char* str = (char*)malloc(20);
	int i,k;
	for(i=init,k=0; i<end; i++,k++)
		str[k]=data[i];
	str[k]='\0';
	return str;
}

/*
    This function is used to make the comparison between the alternatives.
    Data represents all the data in the alternatives.
    Types represents the types of the alternative data.
    Max_mix represents to normalize the values.
    cmp is the result array of the comparison.
    size is the amount of the alternatives in the hierarchy.
    sizeCrit represent the size of sheets, used to jump through the data and types array.
 */
__global__
void acquisitonGKernel(char * data, int* index,  int* types, float* max_min, float* cmp, int size, int sizeCrit){
	//The row is the number of the alternative
	int row = blockIdx.x*size+threadIdx.x;
	//The col is the number of the criteria
	int col = blockIdx.y*size+threadIdx.y;
	int value_alt1_int, value_alt2_int, t;
	float value_alt1_float, value_alt2_float;
	char* sub,*sub2;
	// int k=0;
	// for(int i=0; data[i]!='\0'; i++) {
	//      if(i==index[k]) {printf("|"); k++;}
	//      printf("%c",data[i]);
	// }
	// printf("ALT SIZE: %d , CRIT SIZE %d\n",size, sizeCrit);
	// if(row==0 && col <sizeCrit) { //the thread can do the work
	if(row<size && col <sizeCrit) {                 //the thread can do the work
		int indexRead = row*sizeCrit+col;
		// printf("NEW THREAD ROW %d COL %d SIZE %d SIZECRIT %d\n",row,col,size,sizeCrit);
		value_alt1_int=0;
		value_alt1_float=0.0f;
		t=0;
		// printf("%d # %d # %d # %d\n",indexRead, indexRead+1,index[indexRead],index[indexRead+1]);
		sub=copyStr(data,index[indexRead],index[indexRead+1]);
		if(types[row*sizeCrit+col]==0 || types[row*sizeCrit+col]==2) {
			value_alt1_int=char_to_int(sub);
			// printf("CONVERTED INT %d\n",value_alt1_int);
		}else if(types[row*sizeCrit+col]==1) {
			value_alt1_float=char_to_float(sub);
			// printf("CONVERTED FLOAT %f\n",value_alt1_float);
		}
		for(int alt=0; alt<size; alt++) {
			sub2=copyStr(data,index[alt*sizeCrit+col],index[alt*sizeCrit+(col+1)]);
			// printf("ALTERNATIVE %d - %s # %s\n",alt,sub,sub2);
			value_alt2_int=0;
			value_alt2_float=0.0f;
			//alt*sizeCrit+col will jump over the alternatives to get the same coleria value.
			if(types[alt*sizeCrit+col]==0) {
				t=0;
				value_alt2_int=char_to_int(sub2);
			}else if(types[alt*sizeCrit+col]==1) {
				t=1;
				value_alt2_float=char_to_float(sub2);
			}else if(types[alt*sizeCrit+col]==2) {
				t=2;
				value_alt2_int=char_to_int(sub2);
			}
			// printf("DIVIDED BY %f\n",max_min[col]);
			// printf("SIZE: %d\n",size);
			int indexWrite = row*size*sizeCrit+col*size+alt;
			// int indexWrite = row*size*size/2+alt;
			// printf("WERE I WRITE %d\n",row*size*sizeCrit+col*size+alt);
			//Its used row*size*size/2 to jump correctly in the vector and set the values
			if(t==0) {
				value_alt1_int==value_alt2_int ? cmp[indexWrite]=1 : cmp[indexWrite] = (value_alt1_int-value_alt2_int) / (float) max_min[col];
				// printf("Write in T0 %f\n",cmp[indexWrite]);
			}else if(t==1) {
				// if(value_alt1_float!=value_alt2_float) printf("DIF %f %f\n",value_alt1_float,value_alt2_float);
				value_alt1_float==value_alt2_float ? cmp[indexWrite]=1 : cmp[indexWrite] = (value_alt1_float - value_alt2_float) / (float) max_min[col];
				// printf("Write in T1 %f\n",cmp[indexWrite]);
			}
			else if(t==2) {
				// printf("ALTERNATIVE %d - %s # %s\n",alt,sub,sub2);
				// printf("BOOL %d %d\n",value_alt1_int,value_alt2_int);
				if(value_alt1_int==value_alt2_int) cmp[indexWrite]=1;
				else if(value_alt1_int==1) cmp[indexWrite]=9;
				else if(value_alt1_int==0) cmp[indexWrite]=1/9.0f;
				// printf("Write in T2 %f\n", cmp[indexWrite]);
			}else{
				printf("UNESPECTED VALUE FOR T\n");
			}
		}
	}
}

void AHPG::acquisitionG() {
	//Get the device info
	int devID;
	cudaDeviceProp props;
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&props, devID);
	int block_size = (props.major < 2) ? 16 : 32;

	// Para gerar os pesos das alterntivas, será primeiro captado o MIN e MAX
	// valor das alternativas , após isso será montada as matrizes de cada sheet
	auto alt = this->hierarchy->getAlternatives();
	auto sheets = this->hierarchy->getSheets();
	std::vector<std::string> sheetsNames;
	for_each(sheets.begin(), sheets.end(),
	         [&sheetsNames,
	          this](Hierarchy<VariablesType, WeightType>::Criteria *c) mutable {
		// Get the name off all sheet nodes that aren't boolean
		if (this->hierarchy->getResource()->mBool.count(c->getName()) ==
		    0) {
		        sheetsNames.push_back(c->getName());
		}
	});
	std::map<std::string, float> resultValues;
	std::vector<float>h_resources;
	float min, max;
	for (auto it = sheetsNames.begin(); it != sheetsNames.end(); it++) {
		auto result = std::minmax_element(
			alt.begin(), alt.end(),
			[it](Hierarchy<VariablesType, WeightType>::Alternative *a,
			     Hierarchy<VariablesType, WeightType>::Alternative *b) {
			auto ra = a->getResource();
			auto rb = b->getResource();
			if (ra->mInt.count(*it) > 0) {
			        return ra->mInt[*it] < rb->mInt[*it];
			} else {
			        return ra->mWeight[*it] < rb->mWeight[*it];
			}
		});
		min = max = 0;
		if ((*result.first)->getResource()->mInt.count(*it) > 0) {
			min = (*result.first)->getResource()->mInt[*it];
			max = (*result.second)->getResource()->mInt[*it];
		} else {
			min = (*result.first)->getResource()->mWeight[*it];
			max = (*result.second)->getResource()->mWeight[*it];
		}
		resultValues[*it] = (max - min);
		h_resources.push_back(max-min);
		if (resultValues[*it] == 0) {
			resultValues[*it] = 1;
			h_resources[h_resources.size()-1]=1;
		} else {
			resultValues[*it] /= 9.0;
			h_resources[h_resources.size()-1]=1/9.0;
		}
	}
	// At this point, all the integers and float/WeightType resources  has
	// the max and min values discovered.
	//Prepare the variables to send to the GPU Kernel.
	//Create one vector that get all the map keys of the Data
	//This vector will be represented by
	// V=[key,str(value),key,str(value),...]. All the values are converted to std::string and in the kernel map construction their type are rebuild.
	//To help with the Vector type, use tree types of index, 0 int, 1 float, 2 bool.
	std::string data;//to send vectors to kernel, you must only send the address of the first vector element.
	int dataBytes=0;
	std::vector<int> type;
	std::vector<int> index;
	index.push_back(0);
	int totalResources=0;
	//Iterate through all the alternatives and get their resources.
	for(auto it = alt.begin(); it !=alt.end(); it++) {
		auto resource = (*it)->getResource();
		for(auto const& elem : resource->mInt) {
			//data.push_back(elem.first.c_str());
			int b = std::to_string(elem.second).size();
			data+=std::to_string(elem.second);
			type.push_back(0);
			dataBytes+=b;
			index.push_back(dataBytes);
			totalResources++;
		}
		for(auto const& elem : resource->mWeight) {
			// data.push_back(elem.first.c_str());
			int b = std::to_string(elem.second).size();
			data+=std::to_string(elem.second);
			type.push_back(1);
			dataBytes+=b;
			index.push_back(dataBytes);
			totalResources++;
		}
		for(auto const& elem : resource->mBool) {
			// data.push_back(elem.first.c_str());
			int b = std::to_string(elem.second).size();
			data+=std::to_string(elem.second);
			type.push_back(2);
			dataBytes+=b;
			index.push_back(dataBytes);
			totalResources++;
		}
	}
	totalResources/=alt.size();
	long int resourcesSize = alt.size()*alt.size()*totalResources;
	// totalResources=data.size()/(alt.size()*2);
	std::vector<float> c_result(resourcesSize);
	//All the host data are allocated
	//h_resources
	//Creating the device variables
	dev_array<char> d_data(dataBytes);
	dev_array<int> d_types(type.size());
	dev_array<int> d_index(index.size());
	dev_array<float> d_resources(h_resources.size());
	dev_array<float> d_result(resourcesSize);
	//Alocate the device memory
	//Copy the host variables to device variables
	d_data.set(&data.c_str()[0],dataBytes);
	d_types.set(&type[0],sizeof(int)*type.size());
	d_index.set(&index[0],sizeof(int)*index.size());
	d_resources.set(&h_resources[0], sizeof(float)*h_resources.size());
	//cudaMalloc(d_size, alt.size(), sizeof(int), cudaMemcpyHostToDevice);
	//cudaMalloc(d_sizeCrit, totalResources, sizeof(int), cudaMemcpyHostToDevice);

	// setup execution parameters
	// dim3 threadsPerBlock(2,5);
	dim3 threadsPerBlock(block_size,block_size);
	// dim3 numBlocks( 1,1);
	dim3 numBlocks( ceil(alt.size()/(float)threadsPerBlock.x), ceil(alt.size()/(float)threadsPerBlock.y));
	// printf("\nCALLING KERNEL\n");
	// printf("BlockSize: %d. Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n", block_size,  numBlocks.x, numBlocks.y, numBlocks.z, threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
	// acquisitonGKernel<<< grid, threads >>>(d_data.getData(), d_index.getData(), d_types.getData(), d_resources.getData(),d_result.getData(), alt.size(), totalResources);

	acquisitonGKernel<<< numBlocks, threadsPerBlock >>>(d_data.getData(), d_index.getData(), d_types.getData(), d_resources.getData(),d_result.getData(), alt.size(), totalResources);
	cudaDeviceSynchronize();
	d_result.get(&c_result[0],resourcesSize);
	cudaDeviceSynchronize();
	// std::cout<<"TERMINEI A GPU\n";
	// for(int i=0; i<resourcesSize; i++) {
	//      std::cout<<c_result[i]<<" ";
	// }
	int i=0;
	std::vector<std::vector<std::vector<WeightType> > > allWeights;
	std::vector<std::vector<WeightType> > criteriasWeight;
	std::vector<WeightType> alternativesWeight;
	for (int s=0; s<sheets.size(); s++) {
		criteriasWeight.clear();
		for (int a=0; a<alt.size(); a++) {
			alternativesWeight.clear();
			for (int a2=0; a2<alt.size(); a2++) {
				alternativesWeight.push_back(c_result[i]);
				i++;
			}
			criteriasWeight.push_back(alternativesWeight);
		}
		allWeights.push_back(criteriasWeight);
	}
	// With all the weights calculated, now the weights are set in each edge
	// between the sheets and alternatives
	int aSize = this->hierarchy->getAlternativesCount();
	int size = sheets.size();
	for (int i = 0; i < size; i++) {
		auto edges = sheets[i]->getEdges();
		for (int j = 0; j < aSize; j++) {
			edges[j]->setWeights(allWeights[i][j]);
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
	iterateFuncG(
		&AHPG::checkConsistencyG<Hierarchy<VariablesType, WeightType>::Criteria>,
		hierarchy->getFocus());
}

// void AHPG::run(std::vector<Hierarchy<VariablesType,WeightType>::Alternative*>
// alt){
void AHPG::run(std::vector<Host *> alternatives) {
	if (alternatives.size() == 0) {
		this->conceptionG(true);
	} else {
		Resource *resource = alternatives[0]->getResource();
		for (auto it : resource->mInt) {
			this->hierarchy->addResource(it.first, "int");
		}
		for (auto it : resource->mWeight) {
			this->hierarchy->addResource(it.first, "float");
		}
		for (auto it : resource->mString) {
			this->hierarchy->addResource(it.first, "string");
		}
		for (auto it : resource->mBool) {
			this->hierarchy->addResource(it.first, "bool");
		}
		this->conceptionG(false);
		this->setAlternatives(alternatives);
	}
	this->acquisitionG();
	this->synthesisG();
	// this->consistencyG();
}

std::map<std::string, int> AHPG::getResult() {
	std::map<std::string, int> result;
	WeightType *values = this->hierarchy->getFocus()->getPg();
	std::vector<std::pair<int, WeightType> > alternativesPair;
	for (int i = 0; i < this->hierarchy->getAlternativesCount(); i++) {
		alternativesPair.push_back(std::make_pair(i, values[i]));
	}
	// std::sort(alternativesPair.begin(), alternativesPair.end(),
	//           [](auto &left, auto &right) {
	//      return left.second > right.second;
	// });
	VariablesType name;
	auto alternatives = this->hierarchy->getAlternatives();
	for (unsigned int i = 0; i < (unsigned int)alternativesPair.size(); i++) {
		name = alternatives[alternativesPair[i].first]->getName();
		result[name] = i + 1;
	}
	return result;
}

void AHPG::setAlternatives(std::vector<Host *> alternatives) {
	this->hierarchy->clearAlternatives();
	for (auto it : alternatives) {
		auto a = new Hierarchy<VariablesType, WeightType>::Alternative(it);
		this->hierarchy->addAlternative(a);
	}
	this->hierarchy->addEdgeSheetsAlternatives();
}
