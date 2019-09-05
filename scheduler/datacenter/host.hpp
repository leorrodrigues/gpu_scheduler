#ifndef _HOST_NOT_INCLUDED_
#define _HOST_NOT_INCLUDED_

#include "../main_resources/main_resources_types.hpp"

class Host : public main_resource_t {
public:
Host() : main_resource_t(){
	allocated_resources=0;
	active = false;
	id=0;
	id_g=0;
}

~Host(){
	this->resource.clear();
};

void setResource(std::string resourceName, float value) {
	this->resource[resourceName]->setCapacity(value);
}

void setActive(bool active){
	this->active = active;
}

void setAllocatedResources(unsigned int allocated){
	this->allocated_resources = allocated;
}

void addAllocatedResources(){
	this->allocated_resources++;
}

void removeAllocaredResource(){
	if(this->allocated_resources>=1)
		this->allocated_resources--;
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

bool getActive(){
	return this->active;
}

unsigned int getAllocatedResources(){
	return this->allocated_resources;
}

Host& operator+= (Host& rhs){
	this->allocated_resources += rhs.allocated_resources;
	for(auto it : rhs.resource) {
		(*this->resource[it.first]) += (*it.second); // add the two tree
	}
	return *this;
}

Host& operator-= (Host& rhs){
	this->allocated_resources -= rhs.allocated_resources;
	for(auto it : rhs.resource) {
		(*this->resource[it.first]) -= (*it.second); // subtract the two trees
	}
	return *this;
}

void  addPod(int interval_low, int interval_high, std::map<std::string,std::vector<float> > rhs){
	for(auto const& r : rhs) {
		// this->resource[r.first] -= r.second[2];
		// as the tree represents the total ammount of consumed resources, need to add from the tree to represent the pod insertion
		this->resource[r.first]->insert(interval_low, interval_high, r.second[2]);
	}
	this->allocated_resources++;
}

void removePod(int interval_low, int interval_high, std::map<std::string,std::vector<float> > rhs){
	for(auto const& r : rhs) {
		// this->resource[r.first] += r.second[2];
		// as the tree represents the total ammount of consumed resources, need to remove from the tree to represent the pod removal
		this->resource[r.first]->remove(interval_low, interval_high, r.second[2]);
	}
	this->allocated_resources--;
}

private:
unsigned int allocated_resources;
unsigned int id;
unsigned int id_g;
bool active;
};

#endif
