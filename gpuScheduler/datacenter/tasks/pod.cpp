#include "pod.hpp"

Pod::Pod(){
	this->containers=NULL;
	this->id=0;
	this->host=NULL;

	this->containers_size = 0;
	this->fit=0;
}

Pod::Pod(unsigned int id){
	this->containers=NULL;
	this->id=id;
	this->host=NULL;

	this->containers_size = 0;
	this->fit=0;
}

Pod::~Pod(){
	for(size_t i=0; i<this->containers_size; i++)
		delete(this->containers[i]);
	free(containers);
	this->containers=NULL;
	this->host=NULL;
}

Container** Pod::getContainers(){
	return this->containers;
}

unsigned int Pod::getContainersSize(){
	return this->containers_size;
}

Host* Pod::getHost(){
	return this->host;
}

unsigned int Pod::getFit(){
	return this->fit;
}

void Pod::addContainer(Container* c){
	this->containers_size++;
	this->containers = (Container**) realloc ( this->containers, sizeof(Container*)*this->containers_size);

	this->containers[this->containers_size-1] = c;

	std::map<std::string,float> c_r = c->getResources();

	for(std::map<std::string,float>::iterator it=this->resources.begin(); it!=this->resources.end(); it++) {
		this->resources[it->first]+=c_r[it->first];
	}
}

void Pod::setHost(Host* host){
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
	os<<"\t\t\tepc min: " <<p.resources.at("epc_min")<< "; epc_max: " <<p.resources.at("epc_max")<<"\n";
	os<<"\t\t\tram min: " <<p.resources.at("ram_min")<< "; ram_max: " <<p.resources.at("ram_max")<<"\n";
	os<<"\t\t\tvcpu min: "<<p.resources.at("vcpu_min")<<"; vcpu_max: "<<p.resources.at("vcpu_max")<<"\n";
	os<<"\t}\n";
	return os;
}
