#include "container.hpp"

Container::Container() : Task_Resources(){
	this->links=NULL;
	this->links_size=0;
	this->host_id=0;
	this->host_idg=0;

	this->total_bandwidth_min=0;
	this->total_bandwidth_max=0;
}

Container::~Container(){
	free(this->links);
	this->links=NULL;
}

Link* Container::getLinks(){
	return this->links;
}

float Container::getBandwidthMax(){
	return this->total_bandwidth_max;
}

float Container::getBandwidthMin(){
	return this->total_bandwidth_min;
}

unsigned int Container::getLinksSize(){
	return this->links_size;
}

unsigned int Container::getHostId(){
	return this->host_id;
}

unsigned int Container::getHostIdg(){
	return this->host_idg;
}

void Container::setLink(unsigned int dest, float bandwidth_min, float bandwidth_max){
	this->total_bandwidth_min=bandwidth_min;
	this->total_bandwidth_max=bandwidth_max;

	this->links_size++;
	this->links = (Link*) realloc (this->links,sizeof(Link)*this->links_size);
	this->links[this->links_size-1].destination=dest;
	this->links[this->links_size-1].bandwidth_min=bandwidth_min;
	this->links[this->links_size-1].bandwidth_max=bandwidth_max;
}

void Container::setHostId(unsigned int id){
	this->host_id=id;
}

void Container::setHostIdg(unsigned int id){
	this->host_idg=id;
}

std::ostream& operator<<(std::ostream& os, const Container& c)  {
	os<<"\t\t\t{\n";
	os<<"\t\t\tName: "<<c.id<<"\n";
	os<<"\t\t\tLinks:[\n";
	for(size_t i=0; i<c.links_size; i++) {
		os<<"\t\t\t\tDestination: "<<c.links[i].destination<<"\n";
		os<<"\t\t\t\tBandwidth: "<<c.links[i].bandwidth_min<<" | "<<c.links[i].bandwidth_max<<"\n";
	}
	os<<"\t\t\t]\n";
	for(auto const& [key, val] : c.resources) {
		os<<"\t\t\t"<<key<<"-";
		os<<std::get<0>(val)<<";";
		os<<std::get<1>(val);
		os<<"\n";
	}
	os<<"\t\t\t}\n";
	return os;
}
