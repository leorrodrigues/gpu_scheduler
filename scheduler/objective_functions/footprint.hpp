#ifndef _FOOTPRINT_FUNCTION_
#define _FOOTPRINT_FUNCTION_

#include <string>
#include <map>

namespace ObjectiveFunction {

namespace Footprint {

inline float footprint(Builder *builder, total_resources_t total, int interval_low, int interval_high){
	float temp = builder->getUsedResources(interval_low, interval_high);
	float value = 0;
	for(std::map<std::string,Interval_Tree::Interval_Tree*>::iterator it=total.resource.begin(); it!=total.resource.end(); it++ ) {
		value += total.resource[it->first]->getCapacity();
	}
	value =  temp / value;
	if(value > 1) {
		SPDLOG_ERROR("General Footprint {} higher than 100%", value*100);
		exit(0);
	}
	return (value > 0.0000000001) ? value : 0;
}

inline float vcpu(Builder *builder, total_resources_t total, int interval_low, int interval_high){
	float temp = builder->getUsedResource(interval_low, interval_high, "vcpu");
	float temp2 = total.resource["vcpu"]->getCapacity();
	float value =  temp/temp2;
	if(value > 1) {
		SPDLOG_ERROR("VCPU Footprint {} higher than 100% in time [{},{}]", value*100, interval_low, interval_high);
		exit(0);
	}
	return (value > 0.0000000001) ? value : 0;
}

inline float ram(Builder *builder, total_resources_t total, int interval_low, int interval_high){
	float temp = builder->getUsedResource(interval_low, interval_high, "ram");
	float value =  temp /total.resource["ram"]->getCapacity();
	if(value > 1) {
		SPDLOG_ERROR("RAM Footprint {} higher than 100% in time [{},{}]", value*100, interval_low, interval_high);
		exit(0);
	}
	return (value > 0.0000000001) ? value : 0;
}

//TODO need to update the alocated link
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
