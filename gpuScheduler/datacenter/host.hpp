
#ifndef _Host_NOT_INCLUDED_
#define _Host_NOT_INCLUDED_

#include <string>
#include <map>

#include "tasks/container.hpp"
#include "resource.hpp"

class Host {
public:
Host(){
	active = false;
	allocated_resources = 0;
	name="";
}

Host(std::map<std::string, float> resource) {
	for(auto it : resource) {
		this->resource[it.first]+=it.second;
	}
	active = false;
	allocated_resources = 0;
}

~Host(){
	this->resource.clear();
};

void setResource(std::string name, float v) {
	this->resource[name] = v;
}

void setActive(bool active){
	this->active = active;
}

void setAllocatedResources(int allocated){
	this->allocated_resources = allocated;
}

void addAllocatedResources(){
	this->allocated_resources++;
}

void removeAllocaredResource(){
	this->allocated_resources--;
}

std::map<std::string,float> getResource(){
	return this->resource;
}

std::string getName(){
	return this->name;
}

void setName(std::string name){
	this->name = name;
}

bool getActive(){
	return this->active;
}

int getAllocatedResources(){
	return this->allocated_resources;
}

Host& operator+= (Host& rhs){
	for(auto it : rhs.resource) {
		this->resource[it.first]+=it.second;
	}
	return *this;
}

Host& operator-= (Host& rhs){
	for(auto it : rhs.resource) {
		this->resource[it.first]-=it.second;
	}
	return *this;
}

void  addContainer(Container* rhs){
	int fit = rhs->getFit();
	if(fit==7) { // allocate MAX VCPU AND RAM
		this->resource["memory"]-=rhs->containerResources->ram_max;
		this->resource["vcpu"]-=rhs->containerResources->vcpu_max;
	}else if(fit==8) { // ALLOCATE MAX VCPU AND RAM MIN
		this->resource["memory"]-=rhs->containerResources->ram_min;
		this->resource["vcpu"]-=rhs->containerResources->vcpu_max;
	}else if(fit==10) { // ALLOCATE VCPU MIN AND RAM MAX
		this->resource["memory"]-=rhs->containerResources->ram_max;
		this->resource["vcpu"]-=rhs->containerResources->vcpu_min;
	}else if(fit==11) { // ALLOCATE VCPU AND RAM MIN
		this->resource["memory"]-=rhs->containerResources->ram_min;
		this->resource["vcpu"]-=rhs->containerResources->vcpu_min;
	}
}

void removeContainer(Container* rhs){
	int fit = rhs->getFit();
	if(fit==7) { // allocate MAX VCPU AND RAM
		this->resource["memory"]+=rhs->containerResources->ram_max;
		this->resource["vcpu"]+=rhs->containerResources->vcpu_max;
	}else if(fit==8) { // ALLOCATE MAX VCPU AND RAM MIN
		this->resource["memory"]+=rhs->containerResources->ram_min;
		this->resource["vcpu"]+=rhs->containerResources->vcpu_max;
	}else if(fit==10) { // ALLOCATE VCPU MIN AND RAM MAX
		this->resource["memory"]+=rhs->containerResources->ram_max;
		this->resource["vcpu"]+=rhs->containerResources->vcpu_min;
	}else if(fit==11) { // ALLOCATE VCPU AND RAM MIN
		this->resource["memory"]+=rhs->containerResources->ram_min;
		this->resource["vcpu"]+=rhs->containerResources->vcpu_min;
	}
}


// Host& operator+= (Container& rhs){
//      this->resource["memory"]+=rhs.containerResources->ram_max;
//      this->resource["vcpu"]+=rhs.containerResources->vcpu_max;
//      return *this;
// }
//
// Host& operator-= (Container& rhs){
//      this->resource["memory"]-=rhs.containerResources->ram_max;
//      this->resource["vcpu"]-=rhs.containerResources->vcpu_max;
//      return *this;
// }

private:
std::map<std::string, float> resource;
bool active;
int allocated_resources;     ///< allocated_resources Variable to store the information of how many virtual resources are allocated in this specific host.
std::string name;
};

#endif
