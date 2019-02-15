#ifndef _HOST_NOT_INCLUDED_
#define _HOST_NOT_INCLUDED_

#include "../main_resources/main_resources_types.hpp"

class Host : public main_resource_t {
public:
Host() : main_resource_t(){
	resource["allocated_resources"]=0;
	active = false;
	id=0;
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

void  addPod(std::map<std::string,std::tuple<float,float,bool> > rhs){
	for(auto const& r : rhs)
		if(std::get<2>(r.second))
			this->resource[r.first] -= std::get<1>(r.second);
		else
			this->resource[r.first] -= std::get<0>(r.second);
	this->resource["allocated_resources"]++;
}

void removePod(std::map<std::string,std::tuple<float,float,bool> > rhs){
	for(auto const& r : rhs)
		if(std::get<2>(r.second))
			this->resource[r.first] += std::get<1>(r.second);
		else
			this->resource[r.first] += std::get<0>(r.second);
	this->resource["allocated_resources"]--;
}

private:
bool active;
unsigned int id;
};

#endif
