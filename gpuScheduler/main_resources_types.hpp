#ifndef  _MAIN_RESOURCE_NOT_DEFINED_
#define _MAIN_RESOURCE_NOT_DEFINED_

#include <string>
#include <map>

typedef struct main_resource_t {
	std::map<std::string,float> resource;

	explicit main_resource_t(){
		resource["vcpu"]          = 0;
		resource["ram"]           = 0;
		resource["bandwidth"] = 0;
	}

} main_resource_t;

#endif
