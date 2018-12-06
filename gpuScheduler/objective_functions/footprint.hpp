//FootPrint
// Sum alocado / Sum capacidade total

#ifndef _FOOTPRINT_FUNCTION_
#define _FOOTPRINT_FUNCTION_

namespace ObjectiveFunction {

inline float footprint(consumed_resource_t alocado, total_resources_t total){
	return (alocado.vcpu+alocado.ram)==0 ? 0 : ((alocado.vcpu+alocado.ram)/(total.vcpu+total.ram));
}

inline float vcpu_footprint(consumed_resource_t alocado, total_resources_t total){
	return (alocado.vcpu==0) ? 0 : (alocado.vcpu/total.vcpu);
}

inline float ram_footprint(consumed_resource_t alocado, total_resources_t total){
	return (alocado.ram==0) ? 0 : (alocado.ram/total.ram);
}

}

#endif
