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

	std::map<std::string,float> p_r = pod->getResources();

	if(h_vcpu>=p_r["vcpu_max"]) {
		total+=1;
	}else if(h_vcpu>=p_r["vcpu_min"]) {
		total+=4;
	}else{
		return 0;
	}
	if(h_mem >=p_r["ram_max"]) {
		total+=6;
	} else if(h_mem >=p_r["ram_min"]) {
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
	std::map<std::string,float> t_r = task->getResources();
	if(dc->vcpu - consumed->vcpu >=t_r["vcpu_max"]) {
		total+=1;
	}else if(dc->vcpu - consumed->vcpu >=t_r["vcpu_min"]) {
		total+=4;
	}else{
		return 0;
	}
	if(dc->ram - consumed->ram >=t_r["ram_max"]) {
		// std::cout<<"Memory "<<host->getResource()->mFloat["memory"]<<" AND "<<task->ram_max<<"\n";
		total+=6;
	} else if(dc->ram - consumed->ram >=t_r["ram_min"]) {
		total+=7;
	}else{
		return 0;
	}
	return total;
}

inline void addToConsumed(consumed_resource_t* consumed, std::map<std::string,float> resource, unsigned int fit){
	// The pod was allocated, so the consumed variable has to be updated
	if(fit==7) { // allocate MAX VCPU AND RAM
		consumed->ram  += resource["ram_max"];
		consumed->vcpu += resource["vcpu_max"];
	}else if(fit==8) { // ALLOCATE MAX VCPU AND RAM MIN
		consumed->ram  += resource["ram_min"];
		consumed->vcpu += resource["vcpu_max"];
	}else if(fit==10) { // ALLOCATE VCPU MIN AND RAM MAX
		consumed->ram  += resource["ram_max"];
		consumed->vcpu += resource["vcpu_min"];
	}else if(fit==11) { // ALLOCATE VCPU AND RAM MIN
		consumed->ram  += resource["ram_min"];
		consumed->vcpu += resource["vcpu_min"];
	}
}

inline void subToConsumed(consumed_resource_t* consumed, std::map<std::string,float> resource, unsigned int fit){
	// The pod was allocated, so the consumed variable has to be updated
	if(fit==7) { // allocate MAX VCPU AND RAM
		consumed->ram  -= resource["ram_max"];
		consumed->vcpu -= resource["vcpu_max"];
	}else if(fit==8) { // ALLOCATE MAX VCPU AND RAM MIN
		consumed->ram  -= resource["ram_min"];
		consumed->vcpu -= resource["vcpu_max"];
	}else if(fit==10) { // ALLOCATE VCPU MIN AND RAM MAX
		consumed->ram  -= resource["ram_max"];
		consumed->vcpu -= resource["vcpu_min"];
	}else if(fit==11) { // ALLOCATE VCPU AND RAM MIN
		consumed->ram  -= resource["ram_min"];
		consumed->vcpu -= resource["vcpu_min"];
	}
}

}
#endif
