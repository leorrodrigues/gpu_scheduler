#ifndef _Host_NOT_INCLUDED_
#define _Host_NOT_INCLUDED_

#include <string>
#include <map>

#include "tasks/container.hpp"
#include "resource.hpp"

typedef std::string VariablesType;
typedef float WeightType;

class Host {
public:
Host(){
	resources.mIntSize = 0;
	resources.mFloatSize = 0;
	resources.mStringSize = 0;
	resources.mBoolSize = 0;
	active = false;
	allocated_resources = 0;
}

Host(Resource resource) {
	resources.mIntSize = resource.mIntSize;
	resources.mFloatSize = resource.mFloatSize;
	resources.mStringSize = resource.mStringSize;
	resources.mBoolSize = resource.mBoolSize;
	resources.mInt = resource.mInt;
	resources.mFloat = resource.mFloat;
	resources.mString = resource.mString;
	resources.mBool = resource.mBool;
	active = false;
	allocated_resources = 0;
}

Host(Resource* resource) {
	resources.mIntSize = resource->mIntSize;
	resources.mFloatSize = resource->mFloatSize;
	resources.mStringSize = resource->mStringSize;
	resources.mBoolSize = resource->mBoolSize;
	resources.mInt = resource->mInt;
	resources.mFloat = resource->mFloat;
	resources.mString = resource->mString;
	resources.mBool = resource->mBool;
	active = false;
	allocated_resources = 0;
}

~Host(){
};

void setResource(std::string name, int v) {
	this->resources.mInt[name] = v;
}

void setResource(std::string name, bool v) {
	this->resources.mBool[name] = v;
}

void setResource(std::string name, std::string v) {
	this->resources.mString[name] = v;
}

void setResource(std::string name, WeightType v) {
	this->resources.mFloat[name] = v;
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

Resource *getResource(){
	return &(this->resources);
}

std::string getName(){
	auto it = this->resources.mString.find("name");
	if (it != this->resources.mString.end()) {
		return it->second;
	}
	return "";
}

bool getActive(){
	return this->active;
}

int getAllocatedResources(){
	return this->allocated_resources;
}

Host& operator+= (Host& rhs){
	Resource* resource= this->getResource();
	for(auto it : rhs.getResource()->mInt) {
		if(!resource->mIntSize)
			resource->mInt[it.first]=it.second;
		else
			resource->mInt[it.first]+=it.second;
	}
	for(auto it : rhs.getResource()->mFloat) {
		if(!resource->mFloatSize)
			resource->mFloat[it.first]=it.second;
		else
			resource->mFloat[it.first]+=it.second;
	}
	for(auto it : rhs.getResource()->mBool) {
		if(!resource->mBoolSize)
			resource->mInt[it.first]=it.second;
		else if(it.second) {
			resource->mInt[it.first]=it.second;
		}
	}
	for(auto it : rhs.getResource()->mString) {
		if(!resource->mStringSize)
			resource->mString[it.first]=it.second;
		else if(it.first!="name") {
			resource->mString[it.first]+=it.second;
		}
	}
	return *this;
}

Host& operator+= (Resource& rhs){
	Resource* resource= this->getResource();
	for(auto it : rhs.mInt) {
		resource->mInt[it.first]+=it.second;
	}
	for(auto it : rhs.mFloat) {
		resource->mFloat[it.first]+=it.second;
	}
	return *this;
}

Host& operator-= (Resource& rhs){
	Resource* resource= this->getResource();
	for(auto it : rhs.mInt) {
		resource->mInt[it.first]-=it.second;
	}
	for(auto it : rhs.mFloat) {
		resource->mFloat[it.first]-=it.second;
	}
	return *this;
}

Host& operator+= (Container& rhs){
	Resource* resource= this->getResource();

	resource->mFloat["memory"]+=rhs.containerResources->ram_max;
	resource->mFloat["vcpu"]+=rhs.containerResources->vcpu_max;
	return *this;
}

Host& operator-= (Container& rhs){
	Resource* resource= this->getResource();

	resource->mFloat["memory"]-=rhs.containerResources->ram_max;
	resource->mFloat["vcpu"]-=rhs.containerResources->vcpu_max;
	return *this;
}

private:
Resource resources; ///< Resource variable to store the variables.
bool active;

int allocated_resources; ///< allocated_resources Variable to store the information of how many virtual resources are allocated in this specific host.
};

#endif
