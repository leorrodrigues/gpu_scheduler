#ifndef _FATTREE_NOT_INCLUDED_
#define _FATTREE_NOT_INCLUDED_

#include <vnegpu/generator/fat_tree.cuh>

#include "topologyInterface.cuh"

class FatTree : public Topology {
private:
vnegpu::graph<float> *topology;
std::map<std::string,int> indices;
int indexEdge;
int size;
public:

FatTree(){
	topology=NULL;
	size=0;
}

~FatTree(){
	if(this->topology!=NULL) {
		this->topology->free_graph();
	}
}

void setTopology(){
	topology=vnegpu::generator::fat_tree<float>(this->size);
}

void setSize(int size){
	this->size=size;
}

void setLevel(int level){
}

void setResource(std::map<std::string, float> resource){
	int index=1;
	for(auto it: resource) {
		// for(auto it: resource->mInt) {
		index = this->topology->add_node_variable(it.first);
		this->indices[it.first]=index;
	}
	this->indexEdge=this->topology->add_edge_variable("bandwidth");
	//std::cout<<"\t\tEdge Capacity "<<this->topology->get_num_var_edges()<<"\n";
	//std::cout<<"\t\t\tIndex "<<this->indexEdge<<"\n";
	//std::cout<<"\t\tVariables Capacity "<<this->topology->get_num_var_nodes()<<"\n";
}

void populateTopology(std::vector<Host*> hosts){
	//std::cout<<"Populate topology\n";
	for(int i=0; i<this->topology->get_num_edges(); i++) {
		this->topology->set_variable_edge_undirected(0,i,1); //capacity
		this->topology->set_variable_edge_undirected(1,i,1000); //bandwidth
	}
	int i,k,step;
	i=k=0;
	step=this->size;
	for(auto itHosts: hosts) {
		//std::cout<<"\tNew Host\n";
		itHosts->setIdg(i+k*step);

		std::map<std::string, float> res=itHosts->getResource();
		for(auto it: res) {
			this->topology->set_variable_node(indices[it.first],i+k*step,it.second);
		}
		//For each edge in the host i+k*step
		for(int x=this->topology->get_source_offset(i+k*step); x<this->topology->get_source_offset((i+k*step)+1); x++) {
			this->topology->set_variable_edge(indexEdge,x,res["bandwidth"]);
		}
		//this->topology->set_variable_edge_undirected(indexEdge,(i+k*step),res->mFloat["bandwidth"]);
		//std::cout<<"\t\tSet in "<<i*k<<std::setw(20)<<" edge weight with"<<std::setw(12)<<res->mFloat["bandwidth"]<<"\n";
		i++;
		if(i%this->size==0) k++;
	}
	//this->topology->set_hosts(i);
	//this->topology->check_edges_ids();
}

vnegpu::graph<float>* getGraph(){
	return this->topology;
}

std::string getTopology(){
	return "Fat Tree\n";
}

int getIndexEdge(){
	return this->indexEdge;
}

void listTopology(){
	std::cout<<"\n\nList Topology\n";
	const char *nodeType[] ={"Host node","Switch node","Core switch node"};
	const char *varName[]={"Capacity","vCPU","Bandwidth","Memory","Storage","Security"};
	const char *varEdge[]={"Edge Capacity","Bandwidth"};
	int nVarNodes=this->topology->get_num_var_nodes();
	int nVarEdg=this->topology->get_num_var_edges();
	int nNodes=this->topology->get_num_nodes();
	int nHosts=this->topology->get_hosts();
	int nEdges=this->topology->get_num_edges();
	std::cout<<std::setw(40)<<std::left<<"# Number of Nodes Variables: "<<std::setw(5)<<nVarNodes<<"\n";
	std::cout<<std::setw(40)<<std::left<<"# Number of Edges Variables: "<<std::setw(5)<<nVarEdg<<"\n";
	std::cout<<std::setw(40)<<std::left<<"# Number of Nodes: "<<std::setw(5)<<nNodes<<"\n";
	std::cout<<std::setw(40)<<std::left<<"# Number of Hosts: "<<std::setw(5)<<nHosts<<"\n";
	std::cout<<std::setw(40)<<std::left<<"# Number of Edges: "<<std::setw(5)<<nEdges<<"\n";
	std::cout<<"\t#Host#\n";
	for(int i=0; i<nNodes; i++) {
		std::cout<<"\t\tType: "<<nodeType[this->topology->get_node_type(i)]<<" has the id "<<i<<"\n";
		if(this->topology->get_node_type(i)==0) {// if is a host node
			for(int j=0; j<nVarNodes; j++) {
				std::cout<<"\t\t\t"<<std::setw(20)<<std::left<<varName[j]<<std::setw(5)<<std::right<<this->topology->get_variable_node(j,i)<<"\n";
			}
		}
	}
	std::cout<<"\t#Edge#\n";
	for(int i=0; i<nEdges; i++) {
		std::cout<<"\t\tEdge "<<i<<"\n";
		for(int j=0; j<nVarEdg; j++) {
			std::cout<<"\t\t\t"<<std::setw(20)<<std::left<<varEdge[j]<<std::setw(5)<<std::right<<this->topology->get_variable_edge(j,i)<<"\n";
		}
	}
	std::cout<<"\n\n";
}

};
#endif
