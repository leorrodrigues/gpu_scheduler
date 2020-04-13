#ifndef _HOST_NOT_INCLUDED_
#define _HOST_NOT_INCLUDED_

#include "../main_resources/main_resources_types.hpp"

class Host : public main_resource_t {
public:
Host() : main_resource_t(){
	this->resource["allocated_resources"] = new  Interval_Tree::Interval_Tree();
	id=0;
	id_g=0;
}

~Host(){
	this->resource.clear();
};

void setResource(std::string resourceName, float value) {
	this->resource[resourceName]->setCapacity(value);
}

void setAllocatedResources(unsigned int allocated){
	this->resource["allocated_resources"]->setCapacity(static_cast<float>(allocated));
}

void addAllocatedResources(){
	this->resource["allocated_resources"]->addCapacity(1);
}

void removeAllocaredResource(){
	if(this->resource["allocated_resources"]->getCapacity()>=1)
		this->resource["allocated_resources"]->subtractCapacity(1);
}

std::map<std::string, Interval_Tree::Interval_Tree*> getResource(){
	return this->resource;
}

unsigned int getId(){
	return this->id;
}

unsigned int getIdg(){
	return this->id_g;
}

void setId(unsigned int id){
	this->id = id;
}

void setIdg(unsigned int id_g){
	this->id_g = id_g;
}

unsigned int getAllocatedResources(){
	return static_cast<unsigned int>(this->resource["allocated_resources"]->getCapacity(),0);
}

Host& operator+= (Host& rhs){
	for(auto it : rhs.resource) {
		(*this->resource[it.first]) += (*it.second); // add the two tree
	}
	return *this;
}

Host& operator-= (Host& rhs){
	for(auto it : rhs.resource) {
		(*this->resource[it.first]) -= (*it.second); // subtract the two trees
	}
	return *this;
}

void  addPod(int interval_low, int interval_high, std::map<std::string,std::vector<float> > rhs){
	spdlog::debug("adding a new pod into the host");
	for(auto const& r : rhs) {
		if("allocated_resources" == r.first) continue;
		this->resource[r.first]->insert(interval_low, interval_high, r.second[2]);
	}
	this->resource["allocated_resources"]->addCapacity(1);
}

void removePod(int interval_low, int interval_high, std::map<std::string,std::vector<float> > rhs){
	spdlog::debug("Removing the pod of the host");
	for(auto const& r : rhs) {
		if("allocated_resources" == r.first) continue;
		this->resource[r.first]->remove(interval_low, interval_high, r.second[2]);
	}
	this->resource["allocated_resources"]->subtractCapacity(1);
}

float getUsedResource(int low, int high, std::string name){
	float value = resource[name]->getConsumed(low, high);
	return value;
}

bool isActive(int low, int high){
	for(auto const& r : this->resource) {
		if(r.second->getConsumed(low, high) != 0) {//if any resource is consumed in given time
			return true;
		}
	}
	return false;
}

private:
unsigned int id;
unsigned int id_g;
};

#endif
