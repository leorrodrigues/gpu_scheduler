#ifndef _AHP_NOT_INCLUDED_
#define _AHP_NOT_INCLUDED_

#include "hierarchy.hpp"
#include "multicriteria.hpp"

#include <iomanip>
#include <iterator>
#include <cmath>
#include <cstdlib>
#include <utility>

typedef std::string VariablesType;
typedef float WeightType;

class AHP : public Multicriteria {
private:
std::map<int, WeightType> IR;

typedef typename std::vector<
		Hierarchy<VariablesType, WeightType>::Edge *>::iterator edgeIt;

/*Utility functions*/
void updateAlternatives();
template <typename T> void buildMatrix(T *);
template <typename T> void buildNormalizedmatrix(T *);
template <typename T> void buildPml(T *);
template <typename T> void buildPg(T *);
template <typename T> WeightType partialPg(T *, int);
template <typename T> void deleteMatrix(T*);
template <typename T> void deleteNormalizedMatrix(T*);
template <typename T> void checkConsistency(T *);
void generateContentSchema();

/*Print functions*/
/**WARNING If you want to show all calculated data, you have to call the print function before the next synthesis calculus (i.e., edit the synthesis function to print each step before the next).*/
template <typename T> void printMatrix(T *);
template <typename T> void printNormalizedMatrix(T *);
template <typename T> void printPml(T *);
template <typename T> void printPg(T *);

/*Iterate auxiliar function*/
template <typename F, typename T> void iterateFunc(F, T *);

public:
Hierarchy<VariablesType, WeightType> *hierarchy;
AHP();

void conception(bool);
void acquisition();
void synthesis();
void consistency();
void run(std::vector<Host*> host={});

std::map<std::string,int> getResult();

void setAlternatives(std::vector<Host*>);
};
#endif
