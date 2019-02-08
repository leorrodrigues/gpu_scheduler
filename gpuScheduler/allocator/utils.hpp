#ifndef _UTILS_ALLOCATION_
#define _UTILS_ALLOCATION_

#include <iostream>

namespace Allocator {

inline int checkFit(Host* host, Pod* pod){
	// 7 VCPU AND RAM MAX
	// 8 VCPU MAX RAM MIN
	// 10 VCPU MIN RAM MAX
	// 11 VCPU MIN RAM MIN
	// 0 NOT FIT
	int total=0;

	std::map<std::string, float> a = host->getResource();

	float h_vcpu = host->getResource()["vcpu"];
	float h_mem = host->getResource()["memory"];

	if(h_vcpu>=pod->getVcpuMax()) {
		total+=1;
	}else if(h_vcpu>=pod->getVcpuMin()) {
		total+=4;
	}else{
		return 0;
	}
	if(h_mem >=pod->getRamMax()) {
		total+=6;
	} else if(h_mem >=pod->getRamMin()) {
		total+=7;
	}else{
		return 0;
	}
	return total;
}

inline int checkFit(total_resources_t* dc, consumed_resource_t* consumed, Task* task){
	// 7 VCPU AND RAM MAX
	// 8 VCPU MAX RAM MIN
	// 10 VCPU MIN RAM MAX
	// 11 VCPU MIN RAM MIN
	// 0 NOT FIT
	int total=0;
	if(dc->vcpu - consumed->vcpu >=task->getVcpuMax()) {
		total+=1;
	}else if(dc->vcpu - consumed->vcpu >=task->getVcpuMin()) {
		total+=4;
	}else{
		return 0;
	}
	if(dc->ram - consumed->ram >=task->getRamMax()) {
		// std::cout<<"Memory "<<host->getResource()->mFloat["memory"]<<" AND "<<task->ram_max<<"\n";
		total+=6;
	} else if(dc->ram - consumed->ram >=task->getRamMin()) {
		total+=7;
	}else{
		return 0;
	}
	return total;
}

}
#endif
