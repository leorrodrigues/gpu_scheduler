#ifndef _BCUBE_NOT_INCLUDED_
#define _BCUBE_NOT_INCLUDED_

#include "../thirdparty/vnegpu/generator/bcube.cuh"

#include "topologyInterface.cuh"

class Bcube : public Topology {
private:
vnegpu::graph<float> *topology;
std::map<std::string,int> indices;
std::map<std::string, float> resource;
int indexEdge;
int level;
int size;
public:

Bcube(){
	topology=NULL;
	level=size=0;
}

~Bcube(){
	topology->free_graph();
	topology=NULL;
}

void setTopology(){
	topology=vnegpu::generator::bcube<float>(this->size,this->level);
}

void setSize(int size){
	this->size=size;
}

void setLevel(int level){
	this->level=level;
}

void setResource(std::map<std::string, Interval_Tree::Interval_Tree*> resource){
}

void populateTopology(std::vector<Host*> hosts){

}

vnegpu::graph<float>* getGraph(){
	return this->topology;
}

std::string getTopology(){
	return "BCUBE\n";
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
