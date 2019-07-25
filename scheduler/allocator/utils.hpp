#ifndef _UTILS_ALLOCATION_
#define _UTILS_ALLOCATION_

namespace Allocator {

inline bool checkFit(Host* host, Pod* pod){
	std::map<std::string, float> h_r = host->getResource();

	for(auto r : pod->getResources()) {
		if(h_r[r.first]>=r.second[1]) {
			pod->setFit(r.first, r.second[1]);
		}else if(h_r[r.first]>=r.second[0]) {
			pod->setFit(r.first, h_r[r.first]);
		}else{
			return false;
		}
	}
	return true;
}


inline bool checkFit(total_resources_t* dc, consumed_resource_t* consumed, Task* task){
	for(auto r : task->getResources()) {
		if(dc->resource[r.first] - consumed->resource[r.first]>=r.second[1]) {
			task->setFit(r.first, r.second[1]);
		}else if(dc->resource[r.first] - consumed->resource[r.first] >= r.second[0]) {
			task->setFit(r.first, dc->resource[r.first] - consumed->resource[r.first]);
		}else{
			return false;
		}
	}
	return true;
}

inline void addToConsumed(consumed_resource_t* consumed,Pod* pod){
	for(auto r : pod->getResources()) {
		if(r.second[2]!=0)
			consumed->resource[r.first] += r.second[1];
		else
			consumed->resource[r.first] += r.second[0];
	}
}

inline void subToConsumed(consumed_resource_t* consumed,Pod* pod){
	for(auto r : pod->getResources()) {
		if(r.second[2])
			consumed->resource[r.first] -= r.second[1];
		else
			consumed->resource[r.first] -= r.second[0];
	}
}

}
#endif
