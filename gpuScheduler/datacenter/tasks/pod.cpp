#include "pod.hpp"

Pod::Pod(){
	this->containers=NULL;
	this->id=0;
	this->host=-1;

	this->containers_size = 0;
	this->fit=0;

	this->epc_min=0;
	this->ram_min=0;
	this->vcpu_min=0;
	this->epc_max=0;
	this->ram_max=0;
	this->vcpu_max=0;
}

Pod::Pod(unsigned int id){
	this->containers=NULL;
	this->id=id;
	this->host=-1;

	this->containers_size = 0;
	this->fit=0;

	this->epc_min=0;
	this->ram_min=0;
	this->vcpu_min=0;
	this->epc_max=0;
	this->ram_max=0;
	this->vcpu_max=0;
}

Pod::~Pod(){
	for(size_t i=0; i<this->containers_size; i++)
		delete(this->containers[i]);
	free(containers);
	containers=NULL;
}

Container** Pod::getContainers(){
	return this->containers;
}

unsigned int Pod::getContainersSize(){
	return this->containers_size;
}

unsigned int Pod::getId(){
	return this->id;
}

int Pod::getHost(){
	return this->host;
}

unsigned int Pod::getFit(){
	return this->fit;
}

float Pod::getEpcMin(){
	return this->epc_min;
}

float Pod::getEpcMax(){
	return this->epc_max;
}

float Pod::getRamMin(){
	return this->ram_min;
}

float Pod::getRamMax(){
	return this->ram_max;
}

float Pod::getVcpuMin(){
	return this->vcpu_min;
}

float Pod::getVcpuMax(){
	return this->vcpu_max;
}

void Pod::addContainer(Container* c){
	this->containers_size++;
	this->containers = (Container**) realloc ( this->containers, sizeof(Container*)*this->containers_size);

	this->containres[this->containeres_size-1] = c;

	this->epc_min += c->getEpcMin();
	this->epc_max += c->getEpcMax();
	this->vcpu_min += c->getVcpuMin();
	this->vcpu_max += c->getVcpuMax();
	this->ram_min += c->getRamMin();
	this->ram_max += c->getRamMax();
}

void Pod::setId(unsigned int id){
	this->id=id;
}

void Pod::setHost(int host){
	this->host=host;
}

void Pod::setFit(unsigned int fit){
	this->fit=fit;
}

std::ostream& operator<<(std::ostream& os, const Pod& p)  {
	os<<"\tPod:{\n";
	os<<"\t\tId: "<< p.id<<"\n";
	os<<"\t\tContainers:[\n";
	for(size_t i=0; i<p.containers_size; i++) {
		os<<(*p.containers[i]);
	}
	os<<"\t\t]\n";
	os<<"\t\tTotal Resources\n";
	os<<"\t\t\tepc min: " <<p.epc_min<< "; epc_max: " <<p.epc_max<<"\n";
	os<<"\t\t\tram min: " <<p.ram_min<< "; ram_max: " <<p.ram_max<<"\n";
	os<<"\t\t\tvcpu min: "<<p.vcpu_min<<"; vcpu_max: "<<p.vcpu_max<<"\n";
	os<<"\t}\n";
	return os;
}
