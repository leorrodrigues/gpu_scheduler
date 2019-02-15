#ifndef _TASK_RESOURCES_NOT_INCLUDED_
#define _TASK_RESOURCES_NOT_INCLUDED_

#include <iostream>
#include <tuple>

#include "../../main_resources/main_resources_types.hpp"

class Task_Resources :  main_resource_t {
protected:
std::map<std::string,std::tuple<float,float,bool> > resources;
unsigned int id;

public:
explicit Task_Resources() : main_resource_t(){
	for(auto const&it : this->resource) {
		this->resources[it.first] = std::make_tuple(0,0,false);
	}
	id=0;
}

std::map<std::string,std::tuple<float,float,bool> > getResources(){
	return this->resources;
}


float getResource(std::string key, bool type){
	if(type)
		return std::get<1>(this->resources[key]);
	return std::get<0>(this->resources[key]);
}

bool getFit(std::string key){
	return std::get<2>(this->resources[key]);
}

unsigned int getId(){
	return this->id;
}

void setValue(std::string key, float value, bool type){
	if(type)
		std::get<1>(this->resources[key])=value;
	else
		std::get<0>(this->resources[std::string(key)])=value;
}

void setFit(std::string key, bool fit){
	std::get<2>(this->resources[key])=fit;
}

void setId(unsigned int id){
	this->id=id;
}

};

#endif
