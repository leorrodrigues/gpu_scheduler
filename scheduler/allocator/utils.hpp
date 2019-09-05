#ifndef _UTILS_ALLOCATION_
#define _UTILS_ALLOCATION_

namespace Allocator {

//Check if the pod and its containers fit inside the host
inline bool checkFit(Host* host, Pod* pod, int low, int high){
	std::map<std::string, Interval_Tree::Interval_Tree*> h_r = host->getResource();
	float capacity = 0;
	for(auto r : pod->getResources()) {
		capacity = h_r[r.first]->getMinValueAvailable(low,high);
		if(capacity >= r.second[1]) {
			pod->setFit(r.first, r.second[1]); // stores the maximum capacity asked by the pod
		}else if(capacity >= r.second[0]) {
			pod->setFit(r.first, capacity); //stores the tree capacity
		}else{
			return false;
		}
	}
	return true;
}

//Check if the hole task has less resources than the available
inline bool checkFit(total_resources_t* dc, consumed_resource_t* consumed, Task* task, int low, int high){
	float capacity = 0;
	for(auto r : task->getResources()) {
		capacity = dc->resource[r.first]->getMinValueAvailable(low, high) - consumed->resource[r.first]->getMinValueAvailable(low, high);
		if(capacity >= r.second[1]) {
			task->setFit(r.first, r.second[1]);
		}else if(capacity >= r.second[0]) {
			task->setFit(r.first, capacity);
		}else{
			return false;
		}
	}
	return true;
}

inline void addToConsumed(consumed_resource_t* consumed,Pod* pod, int low, int high){
	for(auto r : pod->getResources()) {
		if(r.second[2]!=0)
			consumed->resource[r.first]->insert(low, high, r.second[1]);
		else
			consumed->resource[r.first]->insert(low, high, r.second[0]);
	}
}

inline void subToConsumed(consumed_resource_t* consumed,Pod* pod, int low, int high){
	for(auto r : pod->getResources()) {
		if(r.second[2])
			consumed->resource[r.first]->remove(low, high, r.second[1]);
		else
			consumed->resource[r.first]->remove(low, high, r.second[0]);
	}
}

}
#endif
