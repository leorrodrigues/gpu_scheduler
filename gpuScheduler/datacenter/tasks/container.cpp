#include "container.hpp"

Container::Container(){
	this->links=NULL;
	this->links_size=0;
}

Container::~Container(){
	free(this->links);
	this->links=NULL;
}

Link* Container::getLinks(){
	return this->links;
}

void Container::setLink(unsigned int dest, float bandwidth_min, float bandwidth_max){
	this->links_size++;
	this->links = (Link*) realloc (this->links,sizeof(Link)*this->links_size);
	this->links[this->links_size-1].destination=dest;
	this->links[this->links_size-1].bandwidth_min=bandwidth_min;
	this->links[this->links_size-1].bandwidth_max=bandwidth_max;
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
	os<<"\t\t\tepc min: "<<c.resources.at("epc_min")<<"; epc_max: "<<c.resources.at("epc_max")<<"\n";
	os<<"\t\t\tram mi: "<<c.resources.at("ram_min")<<"; ram_max: "<<c.resources.at("ram_max")<<"\n";
	os<<"\t\t\tvcpu min: "<<c.resources.at("vcpu_min")<<"; vcpu_max: "<<c.resources.at("vcpu_max")<<"\n";
	os<<"\t\t\t}\n";
	return os;
}
