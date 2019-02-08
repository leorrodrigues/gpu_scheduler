#include "link.hpp"

Link::Link(){
	this->source=0;
	this->destination=0;
	this->name="";
	this->bandwidth_max=0;
	this->bandwidth_min=0;
}

Link::~Link(){
}

void Link::setSource(unsigned int source){
	this->source = source;
}

void Link::setDestination(unsigned int destination){
	this->destination = destination;
}

void Link::setName(const char* name){
	std::string n(name);
	this->name = n;
}

void Link::setBandwidthMax(float bandwidth){
	this->bandiwdth_max = bandwidth;
}

void Link::setBandwidthMin(float bandiwdth){
	this->bandiwdth_min = bandiwdth;
}

unsigned int Link::getSource(){
	return this->source;
}

unsigned int Link::getDestination(){
	return this->destination;
}

const char* Link::getName(){
	return this->name.c_str();
}

float Link::getBandiwdthMax(){
	return this->bandiwdth_max;
}

float Link::getBandiwdthMin(){
	return this->bandiwdth_min;
}

std::ostream& operator<<(std::ostream& os, const Container& c)  {
	os<<"\t\t\t{\n";
	os<<"\t\t\t\tSource: "<<l.source<<"\n";
	os<<"\t\t\t\tDestination: " <<l.destination<<"\n";
	os<<"\t\t\t\tName: "<<l.name<<"\n";
	os<<"\t\t\t\tBandwidth Max: "<<l.bandiwdth_max<<" | Min: "<<l.bandiwdth_min<<"\n";
	os<<"\t\t\t}\n";
	return os;
}
