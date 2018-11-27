#ifndef _DCEL_NOT_INCLUDED_
#define _DCEL_NOT_INCLUDED_

#include <vnegpu/generator/dcell.cuh>

#include "topologyInterface.cuh"

class Dcell : public Topology {
private:
vnegpu::graph<float> *topology;
std::map<std::string,int> indices;
Resource* resource;
int indexEdge;
int level;
int size;
public:

Dcell(){
	topology=NULL;
	level=size=0;
}

void setTopology(){
	topology=vnegpu::generator::dcell<float>(this->size,this->level);
}

void setSize(int size){
	this->size=size;
}

void setLevel(int level){
	this->level=level;
}

void setResource(Resource* resource){
	int index;
	for(auto it: resource->mInt) {
		index=this->topology->add_node_variable(it.first);
		this->indices[it.first]=index;
	}
	for(auto it: resource->mFloat) {
		index=this->topology->add_node_variable(it.first);
		this->indices[it.first]=index;
	}
	for(auto it: resource->mBool) {
		index=this->topology->add_node_variable(it.first);
		this->indices[it.first]=index;
	}
	this->indexEdge=this->topology->add_edge_variable("bandwidth");
}

void populateTopology(std::vector<Host*> hosts){
	std::cout<<"Populate topology\n";
	int i=0;
	for(auto itHosts: hosts) {
		Resource* res=itHosts->getResource();
		for(auto it: res->mInt) {
			this->topology->set_variable_node(indices[it.first],i,(float)it.second);
		}
		for(auto it: res->mFloat) {
			this->topology->set_variable_node(indices[it.first],i,(float)it.second);
		}
		for(auto it: res->mBool) {
			if(it.second==false) {
				this->topology->set_variable_node(indices[it.first],i,(float)1);
			}else{
				this->topology->set_variable_node(indices[it.first],i,(float)0);
			}
		}
		this->topology->set_node_type(i,vnegpu::TYPE_HOST);
		i++;
	}
}

vnegpu::graph<float>* getGraph(){
	return this->topology;
}


std::string getTopology(){
	return "DCELL\n";
}

int getIndexEdge(){
	return this->indexEdge;
}

Resource* getResource(){
	return this->resource;
}

void listTopology(){
	/*for(this->topology->get_hosts()) {

	   }*/
}

};

#endif
