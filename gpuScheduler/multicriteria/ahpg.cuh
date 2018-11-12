#ifndef _AHPG_NOT_INCLUDED_
#define _AHPG_NOT_INCLUDED_

#include "hierarchy/hierarchy_resource.hpp"
#include "hierarchy/hierarchy.hpp"
#include "hierarchy/node.hpp"
#include "hierarchy/edge.hpp"
#include "multicriteria.hpp"

#include "../json.hpp"

#include "ahpg_kernel.cuh"

#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <ctype.h>
#include <cfloat>
#include <string>
#include <cmath>

#include <cuda_runtime.h>
#include <cuda.h>

class AHPG : public Multicriteria {
private:
std::map<int, float> IR;

char path[1024];

/*Utility functions*/
void updateAlternativesG();
void buildMatrixG(Node*);
void buildNormalizedMatrixG(Node*);
void buildPmlG(Node*);
void buildPgG(Node*);
float partialPgG(Node*, int);
void deleteMatrixG(Node*);
void deleteMatrixIG(Node*);
void deleteNormalizedMatrixG(Node*);
void deleteNormalizedMatrixIG(Node*);
void deletePmlG(Node*);
void checkConsistencyG(Node*);
void generateContentSchemaG();

/*Iterate auxiliar function*/
template <typename F> void iterateFuncG(F, Node*);

public:
Hierarchy* hierarchy;

AHPG();
~AHPG();

void setHierarchyG();

void conceptionG(bool);
void acquisitionG();
void synthesisG();
void consistencyG();

/******* Virtual Functions Override*******/
void run(Host** alternatives={}, int size=0);

std::map<int,const char*> getResult();

void setAlternatives(Host** host,int size);
/************************************/

char* strToLowerG(const char*);
void resourcesParserG(genericValue* dataResource);
void hierarchyParserG(genericValue* dataObjective);
void criteriasParserG(genericValue* dataCriteria, Node* p);
void alternativesParserG(genericValue* dataAlternative);
void domParserG(rapidjson::Document* data);

/*Print functions*/
/**WARNING If you want to show all calculated data, you have to call the print function before the next synthesis calculus (i.e., edit the synthesis function to print each step before the next).*/
void printMatrixG(Node*);
void printNormalizedMatrixG(Node*);
void printPmlG(Node*);
void printPgG(Node*);
};
#endif
