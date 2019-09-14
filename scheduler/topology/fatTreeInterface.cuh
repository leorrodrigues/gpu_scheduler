#ifndef _FATTREE_NOT_INCLUDED_
#define _FATTREE_NOT_INCLUDED_

#include <vnegpu/generator/fat_tree.cuh>

#include "topologyInterface.cuh"

class FatTree : public Topology {
private:
vnegpu::graph<float> *topology;
std::map<std::string,int> indices;
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

void setResource(std::map<std::string, Interval_Tree::Interval_Tree*> resource){
	int index=1;
	for(auto it: resource) {
		// for(auto it: resource->mInt) {
		index = this->topology->add_node_variable(it.first);
		this->indices[it.first]=index;
	}
	this->topology->add_edge_variable("bandwidth");
	this->topology->add_edge_variable("connections");

	//std::cout<<"\t\tEdge Capacity "<<this->topology->get_num_var_edges()<<"\n";
	//std::cout<<"\t\t\tIndex "<<"\n";
	//std::cout<<"\t\tVariables Capacity "<<this->topology->get_num_var_nodes()<<"\n";
}

void populateTopology(std::vector<Host*> hosts){
	//std::cout<<"Populate topology\n";
	for(int i=0; i<this->topology->get_num_edges(); i++) {
		this->topology->set_variable_edge_undirected(0,i,1); //capacity
		this->topology->set_variable_edge_undirected(1,i,1000); //bandwidth
		this->topology->set_variable_edge_undirected(2,i,0); //connections
	}
	for(size_t i=0, host_index=0; i< this->topology->get_num_nodes(); i++) {
		if(this->topology->get_node_type(i)==0) {// if is a host node
			hosts[host_index]->setIdg(i);
			std::map<std::string, Interval_Tree::Interval_Tree*> res = hosts[host_index]->getResource();
			for(auto it: res) {
				this->topology->set_variable_node(indices[it.first], i,it.second->getCapacity());
			}
			host_index++;
		}
	}
}

vnegpu::graph<float>* getGraph(){
	return this->topology;
}

std::string getTopology(){
	return "Fat Tree\n";
}

void listTopology(){
	spdlog::info("\n\nList Topology");
	const char *nodeType[] ={"Host node","Switch node","Core switch node"};
	std::vector<std::string> *varName = this->topology->get_var_node_str();
	const char *varEdge[]={"Edge Capacity","Bandwidth"};
	int nVarEdg=this->topology->get_num_var_edges();
	int nNodes=this->topology->get_num_nodes();
	int nHosts=this->topology->get_hosts();
	int nEdges=this->topology->get_num_edges();
	spdlog::info("# Number of Nodes Variables: {}",varName->size());
	spdlog::info("# Number of Edges Variables: {}",nVarEdg);
	spdlog::info("# Number of Nodes: {}",nNodes);
	spdlog::info("# Number of Hosts: {}",nHosts);
	spdlog::info("# Number of Edges: {}",nEdges);
	spdlog::info("\t#Host#");
	for(int i=0; i<nNodes; i++) {
		spdlog::info("\t\tType: {} has the id {}",nodeType[this->topology->get_node_type(i)], i);
		if(this->topology->get_node_type(i)==0) {// if is a host node
			for(int j=0; j<varName->size(); j++) {
				spdlog::info("\t\t\t {} {}",(*varName)[j],this->topology->get_variable_node(j,i));
			}
		}
	}
	spdlog::info("\t#Edge#");
	for(int i=0; i<nEdges; i++) {
		spdlog::info("\t\tEdge ", i);
		for(int j=1; j<nVarEdg; j++) {
			spdlog::info("\t\t\t {} {}", varEdge[j], this->topology->get_variable_edge(j,i));
		}
	}
	spdlog::info("\n");
}

};
#endif
