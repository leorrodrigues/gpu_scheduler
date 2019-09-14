#ifndef _UTILS_ALLOCATION_
#define _UTILS_ALLOCATION_

namespace Allocator {

//Check if the pod and its containers fit inside the host
inline bool checkFit(Host* host, Pod* pod, int low, int high){
	std::map<std::string, Interval_Tree::Interval_Tree*> h_r = host->getResource();
	float capacity = 0;
	for(auto r : pod->getResources()) {
		if(r.first == "allocated_resources") continue;
		h_r[r.first]->show();
		capacity = h_r[r.first]->getMinValueAvailable(low,high);
		if(capacity >= r.second[1]) {
			pod->setFit(r.first, r.second[1]); // stores the maximum capacity asked by the pod
			if(pod->getFit(r.first) != r.second[1]) {
				SPDLOG_ERROR("O VALOR INSERIDO NO POD ESTA DIFERENTE!!!!");
				exit(0);
			}
		} else if(capacity >= r.second[0]) {
			pod->setFit(r.first, capacity); //stores the tree capacity
			if(pod->getFit(r.first) != capacity) {
				SPDLOG_ERROR("O VALOR INSERIDO NO POD ESTA DIFERENTE!!!!");
				exit(0);
			}
		} else {
			return false;
		}
	}
	return true;
}

inline void addToConsumed(consumed_resource_t* consumed, Pod* pod){
	for(auto r : pod->getResources()) {
		(*consumed->resource[r.first]) += r.second[2];
	}
}

inline void subToConsumed(consumed_resource_t* consumed,Pod* pod){
	for(auto r : pod->getResources()) {
		(*consumed->resource[r.first]) -= r.second[2];
	}
}

}
#endif
