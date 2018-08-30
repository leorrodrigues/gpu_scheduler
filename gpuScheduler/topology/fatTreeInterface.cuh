#ifndef _FATTREE_NOT_INCLUDED_
#define _FATTREE_NOT_INCLUDED_

#include <vnegpu/generator/fat_tree.cuh>

#include "topologyInterface.cuh"

class FatTree : public Topology {
private:
vnegpu::graph<float> *topology;
std::map<std::string,int> indices;
Resource* resource;
int indexEdge;
int size;
public:

FatTree(){
	topology=NULL;
	resource=NULL;
	size=0;
}

void setTopology(){
	topology=vnegpu::generator::fat_tree<float>(this->size);
}

void setSize(int size){
	this->size=size;
}

void setLevel(int level){
}

void setResource(Resource* resource){
	//std::cout<<"SetResource\n";
	this->resource=resource;
	int index;
	for(auto it: resource->mInt) {
		index=this->topology->add_node_variable(it.first);
		this->indices[it.first]=index;
	}
	for(auto it: resource->mWeight) {
		index=this->topology->add_node_variable(it.first);
		this->indices[it.first]=index;
	}
	for(auto it: resource->mBool) {
		index=this->topology->add_node_variable(it.first);
		this->indices[it.first]=index;
	}
	this->indexEdge=this->topology->add_edge_variable("bandwidth");
	//std::cout<<"\t\tEdge Capacity "<<this->topology->get_num_var_edges()<<"\n";
	//std::cout<<"\t\t\tIndex "<<this->indexEdge<<"\n";
	//std::cout<<"\t\tVariables Capacity "<<this->topology->get_num_var_nodes()<<"\n";
}

void populateTopology(std::vector<Host*> hosts){
	//std::cout<<"Populate topology\n";
	for(int i=0; i<this->topology->get_num_var_edges(); i++) {
		for(int j=0; j<this->topology->get_num_edges(); j++) {
			this->topology->set_variable_edge_undirected(i,j,1);
		}
	}
	int i,k,step;
	i=k=0;
	step=this->size;
	for(auto itHosts: hosts) {
		//std::cout<<"\tNew Host\n";
		Resource* res=itHosts->getResource();
		for(auto it: res->mInt) {
			this->topology->set_variable_node(indices[it.first],i+k*step,(float)it.second);
			//std::cout<<"\t\tSet "<<std::setw(15)<<it.first<<" with "<<std::setw(15)<<it.second<<"\n";
		}
		for(auto it: res->mWeight) {
			this->topology->set_variable_node(indices[it.first],i+k*step,(float)it.second);
			//std::cout<<"\t\tSet "<<std::setw(15)<<it.first<<" with "<<std::setw(15)<< it.second<<"\n";
		}
		for(auto it: res->mBool) {
			if(it.second==false) {
				this->topology->set_variable_node(indices[it.first],i+k*step,(float)0);
			}else{
				this->topology->set_variable_node(indices[it.first],i+k*step,(float)1);
			}
			//std::cout<<"\t\tSet "<<std::setw(15)<<it.first<<" with "<<std::setw(15) <<it.second<<"\n";
		}
		//For each edge in the host i+k*step
		for(int x=this->topology->get_source_offset(i+k*step); x<this->topology->get_source_offset((i+k*step)+1); x++) {
			this->topology->set_variable_edge(indexEdge,x,res->mWeight["bandwidth"]);
		}
		//this->topology->set_variable_edge_undirected(indexEdge,(i+k*step),res->mWeight["bandwidth"]);
		//std::cout<<"\t\tSet in "<<i*k<<std::setw(20)<<" edge weight with"<<std::setw(12)<<res->mWeight["bandwidth"]<<"\n";
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

Resource* getResource(){
	return this->resource;
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
		std::cout<<"\t\tType: "<<nodeType[this->topology->get_node_type(i)]<<"\n";
		if(this->topology->get_node_type(i)==0) {
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
