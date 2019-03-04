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
	spdlog::debug("\tTuple made");
	spdlog::debug("\tResources size {}",this->resources.size());
	spdlog::debug("\tThe pod has {} containers", this->containers_size);
	for(size_t i=0; i<this->containers_size; i++) {
		spdlog::debug("\t\tGet the values of the container {}",i);
		std::get<0>(this->resources["bandwidth"])+=this->containers[i]->getBandwidthMin();
		std::get<1>(this->resources["bandwidth"])+=this->containers[i]->getBandwidthMax();
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
		spdlog::debug("\t\t\t{} - {}; {}; {}", key, std::get<0>(val),  std::get<1>(val), std::get<2>(val));
	}
	spdlog::debug("\t\t}");
}
