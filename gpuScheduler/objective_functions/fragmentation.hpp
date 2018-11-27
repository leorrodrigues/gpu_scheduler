//Fragmentation
// active servers / total of servers

#ifndef _FRAGMENTATION_FUNCTION_
#define _FRAGMENTATION_FUNCTION_

namespace ObjectiveFunction {

inline float fragmentation(consumed_resource_t consumed,total_resources_t total){
	return consumed.active_servers/ (float)total.servers;
}

inline float fragmentation(int active, int total){
	return active/(float)total;
}

}

#endif
