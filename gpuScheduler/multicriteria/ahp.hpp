#ifndef _AHP_NOT_INCLUDED_
#define _AHP_NOT_INCLUDED_

#include "hierarchy/hierarchy_resource.hpp"
#include "hierarchy/hierarchy.hpp"
#include "hierarchy/node.hpp"
#include "hierarchy/edge.hpp"
#include "multicriteria.hpp"

#include "../json.hpp"

#include <iomanip>
#include <utility>
#include <cstdlib>
#include <cstring>
#include <ctype.h>
#include <cfloat>
#include <string>
#include <cmath>


class AHP : public Multicriteria {
private:
std::map<int, WeightType> IR;


/*Utility functions*/
void updateAlternatives();
void buildMatrix(Node*);
void buildNormalizedmatrix(Node*);
void buildPml(Node*);
void buildPg(Node*);
float partialPg(Node*, int);
void deleteMatrix(Node*);
void deleteNormalizedMatrix(Node*);
void checkConsistency(Node*);
void generateContentSchema();

/*Print functions*/
/**WARNING If you want to show all calculated data, you have to call the print function before the next synthesis calculus (i.e., edit the synthesis function to print each step before the next).*/
void printMatrix(Node*);
void printNormalizedMatrix(Node*);
void printPml(Node*);
void printPg(Node*);

/*Iterate auxiliar function*/
template <typename F> void iterateFunc(F, Node*);

public:
Hierarchy* hierarchy;
AHP();
~AHP();

void setHierarchy();

void conception(bool);
void acquisition();
void synthesis();
void consistency();
void run(Host** host={}, int size=0);

std::map<int, char*> getResult();

void setAlternatives(Host** host,int size);

char* strToLower(const char*);
void resourcesParser(genericValue* dataResource);
void hierarchyParser(genericValue* dataObjective);
void criteriasParser(genericValue* dataCriteria, Node* p);
void alternativesParser(genericValue* dataAlternative);
void domParser(rapidjson::Document* data);
};
#endif
