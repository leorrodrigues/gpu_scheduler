#ifndef _FOOTPRINT_FUNCTION_
#define _FOOTPRINT_FUNCTION_

#include <string>
#include <map>

namespace ObjectiveFunction {

namespace Footprint {

inline float footprint(consumed_resource_t alocado, total_resources_t total, int interval_low, int interval_high){
	float r_t =0, a_t=0;
	for(std::map<std::string,Interval_Tree::Interval_Tree*>::iterator it=alocado.resource.begin(); it!=alocado.resource.end(); it++ ) {
		r_t += total.resource[it->first]->getMinValueAvailable(interval_low, interval_high);
		a_t+= it->second->getMinValueAvailable(interval_low, interval_high);
	}
	float value =  a_t/r_t;
	if(value > 1) {
		SPDLOG_ERROR("General Footprint {} higher than 100%", value*100);
		exit(0);
	}
	return (value > 0.0000000001) ? value : 0;
}

inline float vcpu(consumed_resource_t alocado, total_resources_t total, int interval_low, int interval_high){
	float temp = alocado.resource["vcpu"]->getMinValueAvailable(interval_low, interval_high);
	float temp2 = total.resource["vcpu"]->getMinValueAvailable(interval_low, interval_high);
	float value =  temp/temp2;
	if(value > 1) {
		SPDLOG_ERROR("VCPU Footprint {} higher than 100%", value*100);
		exit(0);
	}
	return (value > 0.0000000001) ? value : 0;
}

inline float ram(consumed_resource_t alocado, total_resources_t total, int interval_low, int interval_high){
	float value =  alocado.resource["ram"]->getMinValueAvailable(interval_low, interval_high) /total.resource["ram"]->getMinValueAvailable(interval_low, interval_high);
	if(value > 1) {
		SPDLOG_ERROR("RAM Footprint {} higher than 100%", value*100);
		exit(0);
	}
	return (value > 0.0000000001) ? value : 0;
}

inline float link(consumed_resource_t alocado, total_resources_t total){
	float value =  alocado.total_bandwidth_consumed /total.total_bandwidth;
	if(value > 1) {
		SPDLOG_ERROR("Link Footprint {} higher than 100%", value*100);
		exit(0);
	}
	return (value > 0.0000000001) ? value : 0;
}

}

}

#endif
