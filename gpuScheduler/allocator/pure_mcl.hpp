#ifndef _MCL_PURE_ALLOCATION_
#define _MCL_PURE_ALLOCATION_

#include <iostream>

#include "utils.hpp"

namespace Allocator {

bool mcl_pure(Builder* builder,  Container* container, std::map<unsigned int, unsigned int> &allocated_task){
	builder->runClustering(builder->getHosts());
	builder->getClusteringResult();
	return true;
}

}
#endif
