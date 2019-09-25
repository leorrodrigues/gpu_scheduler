#ifndef _BEST_FIT_NOT_INCLUDED_
#define _BEST_FIT_NOT_INCLUDED_

#include <algorithm>

#include "../rank.hpp"

#include "../../free.hpp"
#include "../../utils.hpp"

class BestFit : public Rank {
private:
std::vector<Host*> result;
unsigned int *index_result;

struct get_min_element {
	inline bool operator() (Host *host_i, Host *host_j, int low, int high) {
		float cap1 = 0, cap2 = 0;
		// spdlog::info("Bf - Verificar a VCPU do HOST {}",host_i->getId());
		cap1 += host_i->getResource()["vcpu"]->getMinValueAvailable(low, high);
		// spdlog::info("Bf - Verificar a RAM do HOST {}",host_i->getId());
		cap1 += host_i->getResource()["ram"]->getMinValueAvailable(low, high);
		// spdlog::info("Bf - CALCULADO O SOMATORIO");

		// spdlog::info("Bf - Verificar a VCPU do HOST {}",host_j->getId());
		cap2 += host_j->getResource()["vcpu"]->getMinValueAvailable(low, high);
		// spdlog::info("Bf - Verificar a RAM do HOST {}",host_j->getId());
		cap2 += host_j->getResource()["ram"]->getMinValueAvailable(low, high);
		// spdlog::info("Bf - CALCULADO O SOMATORIO");

		return cap1 < cap2;
	}
} get_min_element;

public:
BestFit(){
	index_result = NULL;
}

~BestFit(){
	index_result = NULL;
	result.clear();
}

unsigned int* getResult(unsigned int& size){
	size = result.size();
	index_result = (unsigned int *) calloc (size, sizeof(unsigned int));
	for(int i=0; i < size; i++) {
		index_result[i] = result[i]->getId();
	}
	return index_result;
}

void run(std::vector<Host*> alt, int alt_size, int interval_low, int interval_high){
	result = alt;
	// spdlog::info("BF - Realizar o sort dos vetores no tempo [{},{}]", interval_low, interval_high);
	std::sort(result.begin(), result.end(), std::bind(get_min_element, std::placeholders::_1, std::placeholders::_2, interval_low, interval_high));
}

void readJson(){
}

};

#endif
