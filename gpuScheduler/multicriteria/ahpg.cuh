#ifndef _AHPG_NOT_INCLUDED_
#define _AHPG_NOT_INCLUDED_

#include "hierarchy.hpp"
#include "multicriteria.hpp"

#include "dev_array.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iterator>
#include <utility>

#include <cuda_runtime.h>
#include <cuda.h>

typedef std::string VariablesType;
typedef float WeightType;

class AHPG : public Multicriteria {
private:
public:
std::map<int, WeightType> IR;

typedef typename std::vector<
		Hierarchy<VariablesType, WeightType>::Edge *>::iterator edgeIt;

typedef rapidjson::GenericMember<rapidjson::UTF8<char>, rapidjson::MemoryPoolAllocator<rapidjson::CrtAllocator> > genericValue;
/*Utility functions*/
void updateAlternativesG();
template <typename T> void buildMatrixG(T *);
template <typename T> void buildNormalizedmatrixG(T *);
template <typename T> void buildPmlG(T *);
template <typename T> void buildPgG(T *);
template <typename T> WeightType partialPgG(T *, int);
template <typename T> void deleteMatrixG(T *);
template <typename T> void deleteNormalizedMatrixG(T *);
template <typename T> void checkConsistencyG(T *);
void generateContentSchemaG();

/*Print functions*/
/**WARNING If you want to show all calculated data, you have to call the print
 * function before the next synthesis calculus (i.e., edit the synthesis
 * function to print each step before the next).*/
template <typename T> void printMatrixG(T *);
template <typename T> void printNormalizedMatrixG(T *);
template <typename T> void printPmlG(T *);
template <typename T> void printPgG(T *);

/*Iterate auxiliar function*/
template <typename F, typename T> void iterateFuncG(F, T *);

Hierarchy<VariablesType, WeightType> *hierarchy;
AHPG();

void conceptionG(bool);
void acquisitionG();
void synthesisG();
void consistencyG();

/******* Virtual Functions Override*******/
void run(std::vector<Host *> host = {});

std::map<std::string, int> getResult();

void setAlternatives(std::vector<Host *>);
/************************************/

std::string strToLowerG(std::string s);
void resourcesParserG(genericValue *dataResource);
void hierarchyParserG(genericValue *dataObjective);
template <typename Parent>
void criteriasParserG(genericValue *dataCriteria, Parent p);
void alternativesParserG(genericValue *dataAlternative);
void domParserG(rapidjson::Document *data);
};
#endif
