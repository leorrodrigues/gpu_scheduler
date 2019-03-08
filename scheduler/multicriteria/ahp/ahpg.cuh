#ifndef _AHPG_NOT_INCLUDED_
#define _AHPG_NOT_INCLUDED_

#include "../multicriteria.hpp"
#include "ahpg_kernel.cuh"

#include "../../json.hpp"

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

float *matrix, *normalized_matrix, *pml, *pg;

int *offsets, *edges, hosts_size, hierarchy_size;

public:

AHPG();
~AHPG();

void run(Host** alternatives={}, int size=0);

unsigned int* getResult(unsigned int&);

void setAlternatives(Host** host,int size);

void hierarchyParserG(const rapidjson::Value &hierarchyData);
};
#endif
