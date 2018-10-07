#include "ahp.hpp"

std::string strToLower(std::string s);
void resourcesParser(auto *dataResource, auto ahp);
void hierarchyParser(auto *dataObjective, auto ahp);
template <typename Parent>
void criteriasParser(auto *dataCriteria, Parent p, auto ahp);
void alternativesParser(auto *dataAlternative, auto ahp);
void domParser(rapidjson::Document *data, auto ahp);

AHP::AHP() {
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

std::string strToLower(std::string s) {
	std::transform(s.begin(), s.end(), s.begin(),
	               [](unsigned char c) {
		return std::tolower(c);
	});
	return s;
}

void AHP::updateAlternatives() {
	this->hierarchy->clearAlternatives();
	rapidjson::SchemaDocument alternativesSchema =
		JSON::generateSchema("multicriteria/json/alternativesSchema.json");
	rapidjson::Document alternativesData =
		JSON::generateDocument("multicriteria/json/alternativesDataDefault.json");
	rapidjson::SchemaValidator alternativesValidator(alternativesSchema);
	if (!alternativesData.Accept(alternativesValidator))
		JSON::jsonError(&alternativesValidator);
	domParser(&alternativesData, this);
	this->hierarchy->addEdgeSheetsAlternatives();
}

// Recieve a function address to iterate, used in build matrix, normalized, pml
// and pg functions.
template <typename F, typename T> void AHP::iterateFunc(F f, T *v) {
	std::vector<Hierarchy<VariablesType, WeightType>::Edge *> e = v->getEdges();
	Hierarchy<VariablesType, WeightType>::Criteria *c;
	for (edgeIt it = e.begin(); it != e.end(); it++) {
		c = (*it)->getCriteria();
		if (c != NULL) {
			(this->*f)(c);
		}
	}
}

template <typename T> void AHP::buildMatrix(T *v) {
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
	iterateFunc(&AHP::buildMatrix<Hierarchy<VariablesType, WeightType>::Criteria>,
	            v);
}

template <typename T> void AHP::buildNormalizedmatrix(T *v) {
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
	iterateFunc(&AHP::buildNormalizedmatrix<
			    Hierarchy<VariablesType, WeightType>::Criteria>,
	            v);
}

template <typename T> void AHP::buildPml(T *v) {
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
	iterateFunc(&AHP::buildPml<Hierarchy<VariablesType, WeightType>::Criteria>,
	            v);
}

template <typename T> void AHP::buildPg(T *v) {
	int aSize = this->hierarchy->getAlternativesCount();
	std::vector<Hierarchy<VariablesType, WeightType>::Edge *> e = v->getEdges();
	WeightType *pg = new (std::nothrow) WeightType[aSize];
	for (int i = 0; i < aSize; i++) {
		pg[i] = partialPg(v, i);
	}
	v->setPg(pg);
}

template <typename T> WeightType AHP::partialPg(T *v, int alternative) {
	std::vector<Hierarchy<VariablesType, WeightType>::Edge *> e = v->getEdges();
	int size = e.size();
	Hierarchy<VariablesType, WeightType>::Criteria *c;
	WeightType *pml = v->getPml();
	WeightType partial = 0;
	for (int i = 0; i < size; i++) {
		c = e[i]->getCriteria();
		if (c != NULL) {
			partial += pml[i] * partialPg(c, alternative);
		} else {
			return pml[alternative];
		}
	}
	return partial;
}

template <typename T> void AHP::deleteMatrix(T *v) {
	int size = v->edgesCount();
	WeightType **matrix = v->getMatrix();
	for (int i = 0; i < size; i++)
		delete[] matrix[i];
	delete[] matrix;
	matrix = NULL;
	v->setMatrix(NULL);
	iterateFunc(
		&AHP::deleteMatrix<Hierarchy<VariablesType, WeightType>::Criteria>, v);
}

template <typename T> void AHP::deleteNormalizedMatrix(T *v) {
	int size = v->edgesCount();
	WeightType **nMatrix = v->getNormalizedMatrix();
	for (int i = 0; i < size; i++)
		delete[] nMatrix[i];
	delete[] nMatrix;
	nMatrix = NULL;
	v->setNormalizedMatrix(NULL);
	iterateFunc(&AHP::deleteNormalizedMatrix<
			    Hierarchy<VariablesType, WeightType>::Criteria>,
	            v);
}

template <typename T> void AHP::checkConsistency(T *v) {
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
		std::cout << "ERROR: Criteria: " << v->getName() << " is inconsistent\n";
		std::cout << "RC= " << RC << "\n";
		std::cout<<"SIZE: "<<size<<"\n";
		printMatrix(v);
		printNormalizedMatrix(v);
		printPml(v);
		exit(0);
	}
	iterateFunc(
		&AHP::checkConsistency<Hierarchy<VariablesType, WeightType>::Criteria>,
		v);
}

void AHP::generateContentSchema() {
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

template <typename T> void AHP::printMatrix(T *v) {
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
	iterateFunc(&AHP::printMatrix<Hierarchy<VariablesType, WeightType>::Criteria>,
	            v);
}

template <typename T> void AHP::printNormalizedMatrix(T *v) {
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
	iterateFunc(&AHP::printNormalizedMatrix<
			    Hierarchy<VariablesType, WeightType>::Criteria>,
	            v);
}

template <typename T> void AHP::printPml(T *v) {
	WeightType *pml = v->getPml();
	int tam = v->edgesCount();
	std::cout << "PML of " << v->getName() << "\n";
	for (int i = 0; i < tam; i++) {
		std::cout << std::setfill(' ') << std::setw(10) << pml[i] << " ";
	}
	std::cout << "\n";
	iterateFunc(&AHP::printPml<Hierarchy<VariablesType, WeightType>::Criteria>,
	            v);
}

template <typename T> void AHP::printPg(T *v) {
	WeightType *pg = v->getPg();
	int tam = this->hierarchy->getAlternativesCount();
	std::cout << "PG of " << v->getName() << "\n";
	for (int i = 0; i < tam; i++) {
		std::cout << std::setfill(' ') << std::setw(10) << pg[i] << " ";
	}
	std::cout << "\n";
}

void resourcesParser(auto *dataResource, auto ahp) {
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
		ahp->hierarchy->addResource(variableName, variableType);
	}
}

void hierarchyParser(auto *dataObjective, auto ahp) {
	for (auto &hierarchyObject : dataObjective->value.GetObject()) {
		if (strcmp(hierarchyObject.name.GetString(), "name") == 0) {
			ahp->hierarchy->addFocus(
				strToLower(hierarchyObject.value
				           .GetString())); // create the Focus* in the hierarchy;
		} else if (strcmp(hierarchyObject.name.GetString(), "childs") == 0) {
			criteriasParser(&hierarchyObject, ahp->hierarchy->getFocus(), ahp);
		} else {
			std::cout << "AHP -> Unrecognizable Type\nExiting...\n";
			exit(0);
		}
	}
}

template <typename Parent>
void criteriasParser(auto *dataCriteria, Parent p, auto ahp) {
	std::string name = " ";
	bool leaf = false;
	std::vector<WeightType> weight;
	for (auto &childArray : dataCriteria->value.GetArray()) {
		weight.clear();
		for (auto &child : childArray.GetObject()) {
			const char *n = child.name.GetString();
			if (strcmp(n, "name") == 0) {
				name = strToLower(child.value.GetString());
			} else if (strcmp(n, "leaf") == 0) {
				leaf = child.value.GetBool();
			} else if (strcmp(n, "weight") == 0) {
				for (auto &weightChild : child.value.GetArray()) {
					weight.push_back(weightChild.GetFloat());
				}
			} else if (strcmp(n, "childs") == 0) {
				// at this point, all the criteria variables were read, now the document
				// has the child's of the criteria. To put the childs corretly inside
				// the hierarchy, the criteria node has to be created.
				auto criteria = ahp->hierarchy->addCriteria(name);
				criteria->setLeaf(leaf);
				ahp->hierarchy->addEdge(p, criteria, weight);
				// with the criteria node added, the call recursively the
				// criteriasParser.
				criteriasParser(&child, criteria, ahp);
			}
		}
		if (leaf) {
			auto criteria = ahp->hierarchy->addCriteria(name);
			criteria->setLeaf(leaf);
			ahp->hierarchy->addSheets(criteria);
			ahp->hierarchy->addEdge(p, criteria, weight);
		}
	}
}

void alternativesParser(auto *dataAlternative, auto ahp) {
	for (auto &arrayAlternative : dataAlternative->value.GetArray()) {
		auto alternative = ahp->hierarchy->addAlternative();
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
					name, strToLower(std::string(alt.value.GetString())));
			}
		}
	}
}

void domParser(rapidjson::Document *data, auto ahp) {
	for (auto &m : data->GetObject()) { // query through all objects in data.
		if (strcmp(m.name.GetString(), "resources") == 0) {
			resourcesParser(&m, ahp);
		} else if (strcmp(m.name.GetString(), "objective") == 0) {
			hierarchyParser(&m, ahp);
		} else if (strcmp(m.name.GetString(), "alternatives") == 0) {
			alternativesParser(&m, ahp);
		}
	}
}

void AHP::conception(bool alternativeParser) {
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
		domParser(&resourcesData, this);
		generateContentSchema();
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
	domParser(&hierarchyData, this);
	if (alternativeParser) {
		// The Json Data is valid and can be used to construct the hierarchy.
		rapidjson::SchemaDocument alternativesSchema =
			JSON::generateSchema("multicriteria/json/alternativesSchema.json");
		rapidjson::Document alternativesData = JSON::generateDocument(
			"multicriteria/json/alternativesDataDefault.json");
		rapidjson::SchemaValidator alternativesValidator(alternativesSchema);
		if (!alternativesData.Accept(alternativesValidator))
			JSON::jsonError(&alternativesValidator);
		domParser(&alternativesData, this);
		this->hierarchy->addEdgeSheetsAlternatives();
	}
}

void AHP::acquisition() {
	// Para gerar os pesos das alterntivas, será primeiro captado o MIN e MAX
	// valor das alternativas , após isso será montada as matrizes de cada sheet
	// auto max = this->hierarchy->getResource();
	// auto min = this->hierarchy->getResource();
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
	std::map<std::string, WeightType> resultValues;
	WeightType min, max;
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
		if (resultValues[*it] == 0) {
			resultValues[*it] = 1;
		} else {
			resultValues[*it] /= 9.0;
		}
	}
	// At this point, all the integers and float/WeightType resources  has
	// the max and min values discovered.
	std::vector<std::vector<std::vector<WeightType> > > allWeights;
	std::vector<std::vector<WeightType> > criteriasWeight;
	std::vector<WeightType> alternativesWeight;
	WeightType result;
	for (auto sIt = sheets.begin(); sIt != sheets.end(); sIt++) {
		criteriasWeight.clear();
		for (auto it = alt.begin(); it != alt.end(); it++) {
			alternativesWeight.clear();
			for (auto it2 = alt.begin(); it2 != alt.end(); it2++) {
				if ((*it)->getResource()->mInt.count((*sIt)->getName()) > 0) {
					result = ((*it)->getResource()->mInt[(*sIt)->getName()] -
					          (*it2)->getResource()->mInt[(*sIt)->getName()]) /
					         resultValues[(*sIt)->getName()];
				} else if ((*it)->getResource()->mWeight.count((*sIt)->getName()) > 0) {
					result = ((*it)->getResource()->mWeight[(*sIt)->getName()] -
					          (*it2)->getResource()->mWeight[(*sIt)->getName()]) /
					         resultValues[(*sIt)->getName()];
				} else {
					if ((*it)->getResource()->mBool[(*sIt)->getName()] ==
					    (*it2)->getResource()->mBool[(*sIt)->getName()]) {
						result = 1;
					} else {
						(*it)->getResource()->mBool[(*sIt)->getName()] ? result = 9
						                                                          : result = 1 / 9.0;
					}
				}
				if (result == 0) {
					result = 1;
				} else if (result < 0) {
					result = (-1) / result;
				}
				alternativesWeight.push_back(result);
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

void AHP::synthesis() {
	// 1 - Build the construccd the matrix
	buildMatrix(this->hierarchy->getFocus());
	// printMatrix(this->hierarchy->getFocus());
	// 2 - Normalize the matrix
	buildNormalizedmatrix(this->hierarchy->getFocus());
	// printNormalizedMatrix(this->hierarchy->getFocus());
	// 3 - calculate the PML
	buildPml(this->hierarchy->getFocus());
	// printPml(this->hierarchy->getFocus());
	// 4 - calculate the PG
	buildPg(this->hierarchy->getFocus());
	// printPg(this->hierarchy->getFocus());
	// Print all information
}

void AHP::consistency() {
	iterateFunc(
		&AHP::checkConsistency<Hierarchy<VariablesType, WeightType>::Criteria>,
		hierarchy->getFocus());
}

// void AHP::run(std::vector<Hierarchy<VariablesType,WeightType>::Alternative*>
// alt){
void AHP::run(std::vector<Host *> alternatives) {
	if (alternatives.size() == 0) {
		this->conception(true);
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
		this->conception(false);
		this->setAlternatives(alternatives);
	}
	this->acquisition();
	this->synthesis();
	this->consistency();
	// deleteMatrix(this->hierarchy->getFocus());
	// deleteNormalizedMatrix(this->hierarchy->getFocus());
}

std::map<std::string, int> AHP::getResult() {
	std::map<std::string, int> result;
	WeightType *values = this->hierarchy->getFocus()->getPg();
	std::vector<std::pair<int, WeightType> > alternativesPair;
	for (int i = 0; i < this->hierarchy->getAlternativesCount(); i++) {
		alternativesPair.push_back(std::make_pair(i, values[i]));
	}
	std::sort(alternativesPair.begin(), alternativesPair.end(),
	          [](auto &left, auto &right) {
		return left.second > right.second;
	});
	VariablesType name;
	auto alternatives = this->hierarchy->getAlternatives();
	for (unsigned int i = 0; i < (unsigned int)alternativesPair.size(); i++) {
		name = alternatives[alternativesPair[i].first]->getName();
		result[name] = i + 1;
	}
	return result;
}

void AHP::setAlternatives(std::vector<Host *> alternatives) {
	this->hierarchy->clearAlternatives();
	for (auto it : alternatives) {
		auto a = new Hierarchy<VariablesType, WeightType>::Alternative(it);
		this->hierarchy->addAlternative(a);
	}
	this->hierarchy->addEdgeSheetsAlternatives();
}
