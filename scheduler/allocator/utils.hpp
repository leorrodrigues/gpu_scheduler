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

static bool host_cpu_compare(Host *lhs, Host *rhs){
	std::map<std::string,float> l_r = lhs->getResource();
	std::map<std::string,float> r_r = rhs->getResource();
	return l_r["vcpu"]>r_r["vcpu"];
}

static bool host_ram_compare(Host *lhs, Host *rhs){
	std::map<std::string,float> l_r = lhs->getResource();
	std::map<std::string,float> r_r = rhs->getResource();
	return l_r["ram"]>r_r["ram"];
}

inline bool checkFit(std::vector<Host*> hosts, Task* task){
	std::vector<Host*>::iterator cpu_available = std::max_element(hosts.begin(), hosts.end(),host_cpu_compare);
	std::vector<Host*>::iterator ram_available = std::max_element(hosts.begin(), hosts.end(),host_ram_compare);

	if((*cpu_available)->getResource()["vcpu"] >  task->getResource("vcpu",true)) {
		task->setFit("vcpu",true);
	}else if((*cpu_available)->getResource()["vcpu"] >  task->getResource("vcpu",false)) {
		task->setFit("vcpu",false);
	}else{
		return false;
	}
	if((*ram_available)->getResource()["ram"] > task->getResource("ram",true)) {
		task->setFit("ram",true);
	}else if((*ram_available)->getResource()["ram"] >  task->getResource("ram",false)) {
		task->setFit("ram",false);
	}else{
		return false;
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
