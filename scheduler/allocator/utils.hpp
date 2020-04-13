#ifndef _UTILS_ALLOCATION_
#define _UTILS_ALLOCATION_

namespace Allocator {

//Check if the pod and its containers fit inside the host
inline bool checkFit(Host* host, Pod* pod, int low, int high){
	std::map<std::string, Interval_Tree::Interval_Tree*> h_r = host->getResource();
	for(auto r : pod->getResources()) {
		if(r.first == "allocated_resources") continue;
		if(h_r[r.first]->checkFit(low,high, r.second[1])) {
			pod->setFit(r.first, r.second[1]);
		} else if(r.second[0] != r.second[1] && h_r[r.first]->checkFit(low,high, r.second[0])) {
			pod->setFit(r.first, r.second[0]);
		} else {
			return false;
		}
	}
	return true;
}
}
#endif
