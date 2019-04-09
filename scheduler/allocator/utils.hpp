#ifndef _UTILS_ALLOCATION_
#define _UTILS_ALLOCATION_

namespace Allocator {

inline bool checkFit(Host* host, Pod* pod){
	std::map<std::string, float> h_r = host->getResource();
	for(auto r : pod->getResources()) {
		if(h_r[r.first]>=std::get<1>(r.second)) {
			pod->setFit(r.first, true);
		}else if(h_r[r.first]>=std::get<0>(r.second)) {
			pod->setFit(r.first, false);
		}else{
			return false;
		}
	}
	return true;
}

inline bool checkFit(total_resources_t* dc, consumed_resource_t* consumed, Task* task){
	for(auto r : task->getResources()) {
		if(dc->resource[r.first] - consumed->resource[r.first]>=std::get<1>(r.second)) {
			task->setFit(r.first,true);
		}else if(dc->resource[r.first] - consumed->resource[r.first] >= std::get<0>(r.second) ) {
			task->setFit(r.first,false);
		}else{
			return false;
		}
	}
	return true;
}

inline void addToConsumed(consumed_resource_t* consumed,Pod* pod){
	for(auto r : pod->getResources()) {
		if(std::get<2>(r.second))
			consumed->resource[r.first] += std::get<1>(r.second);
		else
			consumed->resource[r.first] += std::get<0>(r.second);
	}
}

inline void subToConsumed(consumed_resource_t* consumed,Pod* pod){
	for(auto r : pod->getResources()) {
		if(std::get<2>(r.second))
			consumed->resource[r.first] -= std::get<1>(r.second);
		else
			consumed->resource[r.first] -= std::get<0>(r.second);
	}
}

}
#endif
