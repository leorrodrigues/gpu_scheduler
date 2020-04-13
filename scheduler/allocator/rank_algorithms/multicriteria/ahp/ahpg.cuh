#ifndef _AHPG_NOT_INCLUDED_
#define _AHPG_NOT_INCLUDED_

#include "../multicriteria.hpp"
#include "ahpg_kernel.cuh"

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
// the DEVICE PML of the objective and L1 criterias
float *d_pml_obj;
// Final values
float* pg;
unsigned int* hosts_index;

public:

AHPG();
~AHPG();

void run(std::vector<Host*> alt, int alt_size, int interval_low, int interval_high);

unsigned int* getResult(unsigned int&);

void parseAHPG(const rapidjson::Value &hierarchyData);

void readJson();

void setAlternatives(Host** alternatives, int size, int low, int high);
};
#endif
