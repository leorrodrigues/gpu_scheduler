#include "container.hpp"

Container::Container(){
	this->duration=0;
	this->links=NULL;
	this->containerResources = new container_resources_t;
	this->containerResources->name=0;
	this->containerResources->pod=0;
	this->containerResources->epc_min=0;
	this->containerResources->epc_max=0;
	this->containerResources->ram_min=0;
	this->containerResources->ram_max=0;
	this->containerResources->vcpu_min=0;
	this->containerResources->vcpu_max=0;
	this->id=0;
	this->submission=0;
}

void Container::setTask(const char* taskStr){

}

Container::container_resources_t* Container::getResource(){
	return this->containerResources;
}

double Container::getDuration(){
	return this->duration;
}

int Container::getId(){
	return this->id;
}

double Container::getSubmission(){
	return this->submission;
}
