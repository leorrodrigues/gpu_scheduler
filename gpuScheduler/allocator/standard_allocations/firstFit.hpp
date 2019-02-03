#ifndef _FIRST_FIT_NOT_INCLUDED_
#define _FIRST_FIT_NOT_INCLUDED_


#include <iostream>
#include <string>
#include <map>

#include "../../datacenter/tasks/container.hpp"
#include "../../builder.cuh"
#include "../utils.hpp"

#include <limits.h>
#include <queue>

class FirstFit : public Standard {
private:
public:
FirstFit();
~FirstFit();

unsigned int run(Host** host={}, int size=0){

	return INT_MAX;
}

};
#endif
