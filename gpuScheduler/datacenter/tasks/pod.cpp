#include "pod.hpp"

Pod::Pod() : Task_Resources(){
	this->containers=NULL;
	this->id=0;
	this->host=NULL;

	this->containers_size = 0;
}

Pod::Pod(unsigned int id){
	this->containers=NULL;
	this->id=id;
	this->host=NULL;

	this->containers_size = 0;
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

void Pod::addContainer(Container* c){
	this->containers_size++;
	this->containers = (Container**) realloc ( this->containers, sizeof(Container*)*this->containers_size);

	this->containers[this->containers_size-1] = c;

	std::map<std::string,std::tuple<float,float,bool> > c_r = c->getResources();

	float min,max;

	for(auto & [key,val] : this->resources) {
		std::tie(min,max,std::ignore) = c_r[key];
		std::get<0>(val) += min;
		std::get<1>(val) += max;
	}
}

void Pod::setHost(Host* host){
	this->host=host;
	unsigned int id = host->getId();
	for(size_t i=0; i<this->containers_size; i++) {
		this->containers[i].setHostId(id);
	}
}

std::ostream& operator<<(std::ostream& os, const Pod& p)  {
	os<<"\t\t{\n";
	os<<"\t\tId: "<< p.id<<"\n";
	os<<"\t\tContainers:[\n";
	for(size_t i=0; i<p.containers_size; i++) {
		os<<(*p.containers[i]);
	}
	os<<"\t\t]\n";
	os<<"\t\tTotal Resources\n";
	for(auto const& [key, val] : p.resources) {
		os<<"\t\t\t"<<key<<"-";
		os<<std::get<0>(val)<<";";
		os<<std::get<1>(val)<<";";
		os<<std::get<2>(val);
		os<<"\n";
	}
	os<<"\t\t}\n";
	return os;
}
