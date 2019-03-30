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
	float value =  a_t/r_t;
	return (value > 0.0000000001) ? value : 0;
}

inline float vcpu(consumed_resource_t alocado, total_resources_t total){
	float value =  alocado.resource["vcpu"]/total.resource["vcpu"];
	return (value > 0.0000000001) ? value : 0;
}

inline float ram(consumed_resource_t alocado, total_resources_t total){
	float value =  alocado.resource["ram"]/total.resource["ram"];
	return (value > 0.0000000001) ? value : 0;
}

inline float link(consumed_resource_t alocado, total_resources_t total){
	float value =  alocado.resource["bandwidth"] /total.resource["bandwidth"];
	return (value > 0.0000000001) ? value : 0;
}

}

}

#endif
