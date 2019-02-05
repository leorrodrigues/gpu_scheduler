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
	printf("Start Fit\n");
	int total=0;
	printf("Will get the vcpu\n");
	if(host==NULL) printf("CARALHO\n");
	if(host->getResource().empty()) {
		printf("MAP NULL\n");
	}else{
		printf("WTF");
	}
	std::map<std::string, float> a = host->getResource();
	printf("ALOHA\n");
	printf("%f\n", a["vcpu"]);
	printf("IHA\n");
	float h_vcpu = host->getResource()["vcpu"];
	printf("r vcpu get\n");
	float h_mem = host->getResource()["memory"];
	printf("r mem get\n");
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
	printf("End Fit\n");
	return total;
}

inline int checkFit(total_resources_t* dc, consumed_resource_t* consumed, Pod* pod){
	// 7 VCPU AND RAM MAX
	// 8 VCPU MAX RAM MIN
	// 10 VCPU MIN RAM MAX
	// 11 VCPU MIN RAM MIN
	// 0 NOT FIT
	int total=0;
	if(dc->vcpu - consumed->vcpu >=pod->getVcpuMax()) {
		total+=1;
	}else if(dc->vcpu - consumed->vcpu >=pod->getVcpuMin()) {
		total+=4;
	}else{
		return 0;
	}
	if(dc->ram - consumed->ram >=pod->getRamMax()) {
		// std::cout<<"Memory "<<host->getResource()->mFloat["memory"]<<" AND "<<pod->ram_max<<"\n";
		total+=6;
	} else if(dc->ram - consumed->ram >=pod->getRamMin()) {
		total+=7;
	}else{
		return 0;
	}
	return total;
}

}
#endif
