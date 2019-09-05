#ifndef _TOPSIS_NOT_INCLUDED_
#define _TOPSIS_NOT_INCLUDED_

#include "../multicriteria.hpp"

#include "topsis_kernel.cuh"

#include <iterator>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <ctype.h>
#include <cfloat>
#include <string>
#include <cmath>
#include <queue>
#include <map>

#include <cuda_runtime.h>
#include <cuda.h>

class TOPSIS : public Multicriteria {
private:
char path[1024];
float* hosts_value;
int hosts_size;
unsigned int* hosts_index;

public:
TOPSIS();
~TOPSIS();

void getWeights(float*, unsigned int*,std::map<std::string,Interval_Tree::Interval_Tree*>);

void run(Host** alternatives={}, int size=0, int interval_low = 0, int interval_high = 0);

unsigned int* getResult(unsigned int&);

void setAlternatives(Host** alternatives, int size, int low, int high);

void readJson();
};

#endif
