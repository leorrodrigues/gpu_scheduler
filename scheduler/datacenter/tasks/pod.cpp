#include "pod.hpp"

Pod::Pod() : Task_Resources(){
	this->containers=NULL;
	this->id=0;
	this->host=NULL;

	this->containers_size = 0;
}

Pod::Pod(unsigned int id) : Task_Resources() {
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

	std::map<std::string,std::vector<float> > c_r = c->getResources();

	for(auto & [key,val] : this->resources) {
		val[0] += c_r[key][0];
		val[1] += c_r[key][1];
	}
}

void Pod::setHost(Host* host){
	this->host=host;
	unsigned int id = host->getId();
	unsigned int idg = host->getIdg();
	for(size_t i=0; i<this->containers_size; i++) {
		this->containers[i]->setHostId(id);
		this->containers[i]->setHostIdg(idg);
	}
}

void Pod::subId(){
	this->id--;
}

void Pod::updateBandwidth(){
	spdlog::debug("\tUpdateBandwidth Function");
	spdlog::debug("\tThe pod has {} containers", this->containers_size);
	for(size_t i=0; i<this->containers_size; i++) {
		spdlog::debug("\t\tGet the values of the container {}",i);
		this->resources["bandwidth"][0]+=this->containers[i]->getBandwidthMin();
		this->resources["bandwidth"][1]+=this->containers[i]->getBandwidthMax();
		spdlog::debug("\t\tGet");
	}
}

void Pod::print(){
	spdlog::debug("\t\t{");
	spdlog::debug("\t\tId: {}",this->id);
	spdlog::debug("\t\tContainers:[");
	for(size_t i=0; i<this->containers_size; i++) {
		(this->containers[i])->print();
	}
	spdlog::debug("\t\t]");
	spdlog::debug("\t\tTotal Resources");
	for(auto const& [key, val] : this->resources) {
		spdlog::debug("\t\t\t{} - {}; {}; {}", key, val[0],  val[1], val[2]);
	}
	spdlog::debug("\t\t}");
}
