#ifndef _WORST_FIT_NOT_INCLUDED_
#define _WORST_FIT_NOT_INCLUDED_

#include <queue>

#include "../../free.hpp"
#include "../../utils.hpp"

class WorstFit : public Rank {
private:
std::vector<Host*> result;
unsigned int *index_result;

struct get_max_element {
	inline bool operator() (Host *host_i, Host *host_j, int low, int high) {
		float cap1 = 0, cap2 = 0;
		cap1 += host_i->getResource()["vcpu"]->getMinValueAvailable(low, high);
		cap1 += host_i->getResource()["ram"]->getMinValueAvailable(low, high);

		cap2 += host_i->getResource()["vcpu"]->getMinValueAvailable(low, high);
		cap2 += host_i->getResource()["ram"]->getMinValueAvailable(low, high);

		return cap1 > cap2;
	}
} get_max_element;

public:
WorstFit(){
	index_result = NULL;
}
~WorstFit(){
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
	std::sort(result.begin(), result.end(), std::bind(get_max_element, std::placeholders::_1, std::placeholders::_2, interval_low, interval_high));
}

void readJson(){
}
};
#endif
