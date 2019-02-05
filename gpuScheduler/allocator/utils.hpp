#ifndef _UTILS_ALLOCATION_
#define _UTILS_ALLOCATION_

#include <iostream>

namespace Allocator {

inline int checkFit(Host* host, Container* container){
	// 7 VCPU AND RAM MAX
	// 8 VCPU MAX RAM MIN
	// 10 VCPU MIN RAM MAX
	// 11 VCPU MIN RAM MIN
	// 0 NOT FIT
	int total=0;
	double h_vcpu = host->getResource()["vcpu"];
	double h_mem = host->getResource()["memory"];
	if(h_vcpu>=container->containerResources->vcpu_max) {
		// std::cout<<"VCPU "<<host->getResource()->mFloat["vcpu"]<<" AND "<<container->containerResources->vcpu_max;
		total+=1;
	}else if(h_vcpu>=container->containerResources->vcpu_min) {
		total+=4;
	}else{
		return 0;
	}
	if(h_mem >=container->containerResources->ram_max) {
		// std::cout<<"Memory "<<host->getResource()->mFloat["memory"]<<" AND "<<container->containerResources->ram_max<<"\n";
		total+=6;
	} else if(h_mem >=container->containerResources->ram_min) {
		total+=7;
	}else{
		return 0;
	}
	return total;
}

inline int checkFit(total_resources_t* dc, consumed_resource_t* consumed, Container* container){
	// 7 VCPU AND RAM MAX
	// 8 VCPU MAX RAM MIN
	// 10 VCPU MIN RAM MAX
	// 11 VCPU MIN RAM MIN
	// 0 NOT FIT
	int total=0;
	if(dc->vcpu - consumed->vcpu >=container->containerResources->vcpu_max) {
		// std::cout<<"VCPU "<<host->getResource()->mFloat["vcpu"]<<" AND "<<container->containerResources->vcpu_max;
		total+=1;
	}else if(dc->vcpu - consumed->vcpu >=container->containerResources->vcpu_min) {
		total+=4;
	}else{
		return 0;
	}
	if(dc->ram - consumed->ram >=container->containerResources->ram_max) {
		// std::cout<<"Memory "<<host->getResource()->mFloat["memory"]<<" AND "<<container->containerResources->ram_max<<"\n";
		total+=6;
	} else if(dc->ram - consumed->ram >=container->containerResources->ram_min) {
		total+=7;
	}else{
		return 0;
	}
	return total;
}


inline int checkFit(Host* host, Pod* pod){
	// 7 VCPU AND RAM MAX
	// 8 VCPU MAX RAM MIN
	// 10 VCPU MIN RAM MAX
	// 11 VCPU MIN RAM MIN
	// 0 NOT FIT
	int total=0;
	double h_vcpu = host->getResource()["vcpu"];
	double h_mem = host->getResource()["memory"];
	if(h_vcpu>=pod->getVcpuMax()) {
		// std::cout<<"VCPU "<<host->getResource()->mFloat["vcpu"]<<" AND "<<container->containerResources->vcpu_max;
		total+=1;
	}else if(h_vcpu>=pod->getVcpuMin()) {
		total+=4;
	}else{
		return 0;
	}
	if(h_mem>=pod->getRamMax()) {
		// std::cout<<"Memory "<<host->getResource()->mFloat["memory"]<<" AND "<<container->containerResources->ram_max<<"\n";
		total+=6;
	} else if(h_mem>=pod->getRamMin()) {
		total+=7;
	}else{
		return 0;
	}
	return total;
}

}
#endif
