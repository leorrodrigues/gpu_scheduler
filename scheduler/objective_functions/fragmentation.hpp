//Fragmentation
// active servers / total of servers

#ifndef _FRAGMENTATION_FUNCTION_
#define _FRAGMENTATION_FUNCTION_

namespace ObjectiveFunction {

namespace Fragmentation {

inline float datacenter(consumed_resource_t consumed,total_resources_t total){
	return consumed.active_servers/ (float)total.servers;
}

inline float datacenter(unsigned int active, unsigned int total){
	return active/(float)total;
}

inline float link(consumed_resource_t consumed, total_resources_t total){
	return consumed.active_links/ (float) total.links;
}

inline float link(unsigned int active, unsigned int total){
	return active/(float)total;
}

}
}

#endif
