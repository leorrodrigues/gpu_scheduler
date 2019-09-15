//Fragmentation
// active servers / total of servers

#ifndef _FRAGMENTATION_FUNCTION_
#define _FRAGMENTATION_FUNCTION_

#include "../main_resources/types.hpp"

namespace ObjectiveFunction {

namespace Fragmentation {

inline float datacenter(consumed_resource_t consumed,total_resources_t total){
	float value = (consumed.active_servers/ (float)total.servers);
	if(value > 1) {
		SPDLOG_ERROR("DC Fragmentation {} higher than 100%", value*100);
		exit(0);
	}       return (value >= 0.0000000001) ? value : 0;
}

inline float datacenter(unsigned int active, unsigned int total){
	float value =  active/(float)total;
	if(value > 1) {
		SPDLOG_ERROR("DC Fragmentation {} higher than 100%", value*100);
		exit(0);
	}
	return (value >= 0.0000000001) ? value : 0;
}

inline float link(consumed_resource_t consumed, total_resources_t total){
	float value =  consumed.active_links/ (float) total.links;
	if(value > 1) {
		SPDLOG_ERROR("Link Fragmentation {} higher than 100%", value*100);
		exit(0);
	}
	return (value >= 0.0000000001) ? value : 0;

}

inline float link(unsigned int active, unsigned int total){
	float value =  active/(float)total;
	if(value > 1) {
		SPDLOG_ERROR("Link Fragmentation {} higher than 100%", value*100);
		exit(0);
	}
	return (value >= 0.0000000001) ? value : 0;
}

}
}

#endif
