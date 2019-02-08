#include "container.hpp"

Container::Container(){
	this->links=NULL;
	this->links_size=0;

	this->name=0;
	this->host=-1;
	this->epc_min=0;
	this->epc_max=0;
	this->ram_min=0;
	this->ram_max=0;
	this->vcpu_min=0;
	this->vcpu_max=0;
}

Container::~Container(){
	for(size_t i=0; i<this->links_size; i++) {
		delete(this->links[i]);
	}
	free(this->links);
	this->links=NULL;
}

unsigned int Container::getHost(){
	return this->host;
}

unsigned int Container::getName(){
	return this->name;
}

float Container::getEpcMin(){
	return this->epc_min;
}

float Container::getEpcMax(){
	return this->epc_max;
}

float Container::getRamMin(){
	return this->ram_min;
}

float Container::getRamMax(){
	return this->ram_max;
}

float Container::getVcpuMin(){
	return this->vcpu_min;
}

float Container::getVcpuMax(){
	return this->vcpu_max;
}

void Container::setPod(unsigned int pod){
	this->pod=pod;
}

void Container::setName(unsigned int name){
	this->name=name;
}

void Container::setEpcMin(float epcMin){
	this->epc_min=epcMin;
}

void Container::setEpcMax(float epcMax){
	this->epc_max=epcMax;
}

void Container::setRamMin(float ramMin){
	this->ram_min=ramMin;
}

void Container::setRamMax(float ramMax){
	this->ram_max=ramMax;
}

void Container::setVcpuMin(float vcpuMin){
	this->vcpu_min = vcpuMin;
}

void Container::setVcpuMax(float vcpuMax){
	this->vcpu_max = vcpuMax;
}

std::ostream& operator<<(std::ostream& os, const Container& c)  {
	os<<"\t\t{\n";
	os<<"\t\t\tName: "<<c.name<<"\n";
	os<<"\t\t\tHost: " <<c.host<<"\n";
	os<<"\t\t\tLinks:[\n";
	for(size_t i=0; i<this->links_size; i++) {
		os<<(*this->links[i])<<"\n";
	}
	os<<"\t\t\t]\n";
	os<<"\t\t\tepc min: "<<c.epc_min<<"; epc_max: "<<c.epc_max<<"\n";
	os<<"\t\t\tram min: "<<c.ram_min<<"; ram_max: "<<c.ram_max<<"\n";
	os<<"\t\t\tvcpu min: "<<c.vcpu_min<<"; vcpu_max: "<<c.vcpu_max<<"\n";
	os<<"\t\t}\n";
	return os;
}
