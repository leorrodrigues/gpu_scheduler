#ifndef _DCEL_NOT_INCLUDED_
#define _DCEL_NOT_INCLUDED_

#include <vnegpu/generator/dcell.cuh>

#include "topologyInterface.cuh"

class Dcell : public Topology {
private:
vnegpu::graph<float>* topology;
std::map<std::string,int> indices;
std::map<std::string, float> resource;
int indexEdge;
int level;
int size;
public:

Dcell(){
	topology=NULL;
	level=size=0;
}

~Dcell(){
	this->topology->free_graph();
	this->topology=NULL;
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

void setResource(std::map<std::string, Interval_Tree::Interval_Tree*> resource){
	int index;
	for(auto it: resource) {
		index=this->topology->add_node_variable(it.first);
		this->indices[it.first]=index;
	}
	this->indexEdge=this->topology->add_edge_variable("bandwidth");
}

void populateTopology(std::vector<Host*> hosts){
	std::cout<<"Populate topology\n";
	int i=0;
	for(auto itHosts: hosts) {
		std::map<std::string, Interval_Tree::Interval_Tree*> res=itHosts->getResource();
		for(auto it: res) {
			this->topology->set_variable_node(indices[it.first],i,it.second->getCapacity());
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

std::map<std::string, float> getResource(){
	return this->resource;
}

void listTopology(){
	/*for(this->topology->get_hosts()) {

	   }*/
}

};

#endif
