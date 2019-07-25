#ifndef _TASK_RESOURCES_NOT_INCLUDED_
#define _TASK_RESOURCES_NOT_INCLUDED_

#include <iostream>
#include <tuple>

#include "../../main_resources/main_resources_types.hpp"

class Task_Resources :  main_resource_t {
protected:
std::map<std::string, std::vector<float> > resources;
std::map<std::string,float> total_allocated;
std::map<std::string,float> total_max;
unsigned int id;

public:
explicit Task_Resources() : main_resource_t(){
	std::vector<float> empty_f (3,0);
	for(auto const&it : this->resource) {
		this->resources[it.first] = empty_f;
		this->total_max[it.first] = 0;
		this->total_allocated[it.first] = 0;
	}
	id=0;
	spdlog::debug("\t\tTask Resource has {} members",this->resources.size());
}

std::map<std::string, std::vector<float> > getResources(){
	return this->resources;
}


float getResource(std::string key, bool type){
	if(type)
		return this->resources[key][1];
	return this->resources[key][0];
}

bool getFit(std::string key){
	return this->resources[key][2]==0 ? false : true;
}

unsigned int getId(){
	return this->id;
}

float getMaxResource(std::string key){
	return this->resources[key][1];
}

float getTotalAllocated(std::string key){
	return this->total_allocated[key];
}

void setValue(std::string key, float value, bool type){
	if(type) {
		this->resources[key][1]=value;
		this->total_max[key] = value;
	}else{
		this->resources[key][0]=value;
	}
}

void addValue(std::string key, float value, bool type){
	if(type) {
		this->resources[key][1]=+value;
		this->total_max[key] =+value;
	}else{
		this->resources[key][0]=+value;
	}
}

void setFit(std::string key, float value){
	this->resources[key][2] = value;
	this->total_allocated[key] = value;
}

void setId(unsigned int id){
	this->id=id;
}

};

#endif
