#ifndef _HOST_NOT_INCLUDED_
#define _HOST_NOT_INCLUDED_

#include <string>
#include <map>

class Host {
public:
Host(){
	active = false;
	id=0;
}

Host(std::map<std::string, float> resource) {
	for(auto it : resource) {
		this->resource[it.first]+=it.second;
	}
	active = false;
}

~Host(){
	this->resource.clear();
};

void setResource(std::string resourceName, float value) {
	this->resource[resourceName] = value;
}

void setActive(bool active){
	this->active = active;
}

void setAllocatedResources(unsigned int allocated){
	this->resource["allocated_resources"] = allocated;
}

void addAllocatedResources(){
	this->resource["allocated_resources"]++;
}

void removeAllocaredResource(){
	if(this->resource["allocated_resources"]>=1)
		this->resource["allocated_resources"]--;
}

std::map<std::string,float> getResource(){
	return this->resource;
}

unsigned int getId(){
	return this->id;
}

void setId(unsigned int id){
	this->id = id;
}

bool getActive(){
	return this->active;
}

unsigned int getAllocatedResources(){
	return this->resource["allocated_resources"];
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

void  addPod(std::map<std::string,float> rhs, unsigned int fit){
	if(fit==7) { // allocate MAX VCPU AND RAM
		this->resource["memory"]-=rhs["ram_max"];
		this->resource["vcpu"]-=rhs["ram_max"];
	}else if(fit==8) { // ALLOCATE MAX VCPU AND RAM MIN
		this->resource["memory "]-=rhs["ram_min"];
		this->resource["vcpu "]-=rhs["vcpu_max"];
	}else if(fit==10) { // ALLOCATE VCPU MIN AND RAM MAX
		this->resource["memory "]-=rhs["ram_max"];
		this->resource["vcpu "]-=rhs["vcpu_min"];
	}else if(fit==11) { // ALLOCATE VCPU AND RAM MIN
		this->resource["memory "]-=rhs["ram_min"];
		this->resource["vcpu "]-=rhs["vcpu_min"];
	}
}

void removePod(std::map<std::string,float> rhs, unsigned int fit){
	if(fit==7) { // allocate MAX VCPU AND RAM
		this->resource["memory "]+=rhs["ram_max"];
		this->resource["vcpu "]+=rhs["ram_max"];
	}else if(fit==8) { // ALLOCATE MAX VCPU AND RAM MIN
		this->resource["memory"]+=rhs["ram_min"];
		this->resource["vcpu"]+=rhs["vcpu_max"];
	}else if(fit==10) { // ALLOCATE VCPU MIN AND RAM MAX
		this->resource["memory"]+=rhs["ram_max"];
		this->resource["vcpu"]+=rhs["vcpu_min"];
	}else if(fit==11) { // ALLOCATE VCPU AND RAM MIN
		this->resource["memory"]+=rhs["ram_min"];
		this->resource["vcpu"]+=rhs["vcpu_min"];
	}
}


// Host& operator+= (Container& rhs){
//      this->resource["memory"]+=rhs.ram_max;
//      this->resource["vcpu"]+=rhs.vcpu_max;
//      return *this;
// }
//
// Host& operator-= (Container& rhs){
//      this->resource["memory"]-=rhs.ram_max;
//      this->resource["vcpu"]-=rhs.vcpu_max;
//      return *this;
// }

private:
std::map<std::string, float> resource;
bool active;
unsigned int id;
};

#endif
