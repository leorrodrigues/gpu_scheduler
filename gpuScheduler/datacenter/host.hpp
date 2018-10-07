#ifndef _Host_NOT_INCLUDED_
#define _Host_NOT_INCLUDED_

#include <string>
#include <map>

#include "resource.hpp"

typedef std::string VariablesType;
typedef float WeightType;

class Host {
public:
Host(){
	resources.mIntSize = 0;
	resources.mWeightSize = 0;
	resources.mStringSize = 0;
	resources.mBoolSize = 0;
}

Host(Resource resource) {
	resources.mIntSize = resource.mIntSize;
	resources.mWeightSize = resource.mWeightSize;
	resources.mStringSize = resource.mStringSize;
	resources.mBoolSize = resource.mBoolSize;
	resources.mInt = resource.mInt;
	resources.mWeight = resource.mWeight;
	resources.mString = resource.mString;
	resources.mBool = resource.mBool;
}

Host(Resource* resource) {
	resources.mIntSize = resource->mIntSize;
	resources.mWeightSize = resource->mWeightSize;
	resources.mStringSize = resource->mStringSize;
	resources.mBoolSize = resource->mBoolSize;
	resources.mInt = resource->mInt;
	resources.mWeight = resource->mWeight;
	resources.mString = resource->mString;
	resources.mBool = resource->mBool;
}

~Host();

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
	this->resources.mWeight[name] = v;
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

Host& operator+= (Host& rhs){
	Resource* resource= this->getResource();
	for(auto it : rhs.getResource()->mInt) {
		if(!resource->mIntSize)
			resource->mInt[it.first]=it.second;
		else
			resource->mInt[it.first]+=it.second;
	}
	for(auto it : rhs.getResource()->mWeight) {
		if(!resource->mWeightSize)
			resource->mWeight[it.first]=it.second;
		else
			resource->mWeight[it.first]+=it.second;
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

private:
Resource resources; ///< Resource variable to store the variables.
};

#endif
