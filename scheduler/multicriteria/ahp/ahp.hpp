#ifndef _AHP_NOT_INCLUDED_
#define _AHP_NOT_INCLUDED_

#include "hierarchy/hierarchy_resource.hpp"
#include "hierarchy/hierarchy.hpp"
#include "hierarchy/node.hpp"
#include "hierarchy/edge.hpp"
#include "../multicriteria.hpp"

#include <cstdlib>
#include <cstring>
#include <ctype.h>
#include <unistd.h>
#include <cfloat>
#include <string>
#include <cmath>
#include <queue>

#include <chrono>

class AHP : public Multicriteria {
private:
std::map<int, float> IR;

char path[1024];

/*Utility functions*/
void updateAlternatives();
void buildMatrix(Node*);
void buildNormalizedmatrix(Node*);
void buildPml(Node*);
void buildPg(Node*);
float partialPg(Node*, int);
void deleteMatrix(Node*);
void deleteNormalizedMatrix(Node*);
void deletePml(Node*);
void checkConsistency(Node*);

/*Iterate auxiliar function*/
template <typename F> void iterateFunc(F, Node*);

public:
Hierarchy* hierarchy;
AHP();
~AHP();

void setHierarchy();

void conception();
void acquisition();
void synthesis();
void consistency();
void run(Host**host={}, int size=0, int interval_low = 0, int interval_high = 0);

unsigned int* getResult(unsigned int&);

void setAlternatives(Host** alternatives,int size, int low, int high);

void readJson();

char* strToLower(const char*);
void hierarchyParser(const rapidjson::Value &hierarchyData);

/*Print functions*/
/**WARNING If you want to show all calculated data, you have to call the print function before the next synthesis calculus (i.e., edit the synthesis function to print each step before the next).*/
void printMatrix(Node*);
void printNormalizedMatrix(Node*);
void printPml(Node*);
void printPg(Node*);
};
#endif
