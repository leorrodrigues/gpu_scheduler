//FootPrint
// Sum alocado / Sum capacidade total

#ifndef _FOOTPRINT_FUNCTION_
#define _FOOTPRINT_FUNCTION_

#include <string>
#include <map>

namespace ObjectiveFunction {

namespace Footprint {

inline float footprint(consumed_resource_t alocado, total_resources_t total){
	float r_t =0, a_t=0;
	for(std::map<std::string,float>::iterator it=alocado.resource.begin(); it!=alocado.resource.end(); it++ ) {
		r_t += total.resource[it->first];
		a_t+= it->second;
	}
	return a_t/r_t;
}

inline float vcpu(consumed_resource_t alocado, total_resources_t total){
	return alocado.resource["vcpu"]/total.resource["vcpu"];
}

inline float ram(consumed_resource_t alocado, total_resources_t total){
	return alocado.resource["ram"]/total.resource["ram"];
}

inline float link(consumed_resource_t alocado, total_resources_t total){
	return alocado.resource["bandwidth"]/total.resource["bandwidth"];
}

}

}

#endif
