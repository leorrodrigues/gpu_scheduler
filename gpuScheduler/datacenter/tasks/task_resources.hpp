#ifndef _TASK_RESOURCES_NOT_INCLUDED_
#define _TASK_RESOURCES_NOT_INCLUDED_

#include "map"
#include "string"

class Task_Resources {
protected:
std::map<std::string,float> resources;
unsigned int id;

public:
explicit Task_Resources(){
	resources["epc_min"]=0;
	resources["ram_min"]=0;
	resources["vcpu_min"]=0;
	resources["epc_max"]=0;
	resources["ram_max"]=0;
	resources["vcpu_max"]=0;
	id=0;
}

std::map<std::string,float> getResources(){
	return this->resources;
}

float getResource(std::string key){
	return this->resources[key];
}

float getResource(const char* key){
	return this->resources[std::string(key)];
}

unsigned int getId(){
	return this->id;
}

void setValue(std::string key, float value){
	this->resources[key]=value;
}

void setValue(const char* key, float value){
	this->resources[std::string(key)]=value;
}

void setId(unsigned int id){
	this->id=id;
}

};

#endif
