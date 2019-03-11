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
#include <queue>
#include <cmath>

#include <cuda_runtime.h>
#include <cuda.h>

class AHPG : public Multicriteria {
private:
std::map<int, float> IR;
char path[1024];

// variables updated through hierarchyParser function
// The total ammount of alternatives in the AHPG
unsigned int hosts_size;
// The total ammount of criterias (all the criterias are leaf, just one level of hierarchy)
unsigned int criteria_size;
// Weights of the edges between the objective function and all the criterias in level 1
float *edges_values;
// Final values
float* hosts_value;
unsigned int* hosts_index;

public:

AHPG();
~AHPG();

void run(Host** alternatives={}, int size=0);

unsigned int* getResult(unsigned int&);

void parseAHPG(const rapidjson::Value &hierarchyData);

void readJson();

void setAlternatives(Host**host, int size);
};
#endif
