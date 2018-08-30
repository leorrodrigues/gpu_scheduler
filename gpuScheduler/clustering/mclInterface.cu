#include "mclInterface.cuh"

#include <iostream>

MCLInterface::MCLInterface(){
}

void MCLInterface::run(Topology* topology){
	vnegpu::graph<float>* dataCenter=topology->getGraph();
	dataCenter->update_gpu();
	vnegpu::algorithm::mcl(dataCenter,topology->getIndexEdge(),2,1.2,0);

	dataCenter->update_cpu();
}


//std::map<int,Host*> MCLInterface::getResult(Topology* topology,std::vector<Host*> hosts){
std::vector<Host*> MCLInterface::getResult(Topology* topology,std::vector<Host*> hosts){
	vnegpu::graph<float>* cluster=topology->getGraph();
	std::map<int,Host*> groups;
	std::vector<Host*> vGroups;
	int* graph_groups=cluster->get_group_ptr();
	for(int i=0; i<cluster->get_hosts(); i++) {
		if ( groups.find(graph_groups[i]) == groups.end() ) {
			//if the group key don't exists
			//Host *host=new Host(topology->getResource());
			//std::cout<<host->getResource()->mIntSize<<"\n";
			//groups[i]=host;
			groups[graph_groups[i]]=new Host(topology->getResource());
			groups[graph_groups[i]]->setResource("name",std::to_string(i));//or ID
		}
		(*groups[graph_groups[i]])+=(*hosts[i]); // the group ID has the node i
	}
	//a vazao (edge) do grupo é o menor valor destre os    hosts do grupo.
	for(auto it: groups) {
		vGroups.push_back(it.second);
	}
	return vGroups;
}

void MCLInterface::listGroups(Topology* topology){
	std::cout<<"\n\nMCL Clustering Topology\n";
	vnegpu::graph<float>* cluster=topology->getGraph();
	const char *nodeType[] ={"Host node","Switch node","Core switch node"};
	//const char *varEdge[]={"Edge Capacity","Bandwidth"};
	int nVarNodes=cluster->get_num_var_nodes();
	int nVarEdg=cluster->get_num_var_edges();
	int nNodes=cluster->get_num_nodes();
	int nHosts=cluster->get_hosts();
	int nEdges=cluster->get_num_edges();
	std::cout<<std::setw(40)<<std::left<<"# Number of Nodes Variables: "<<std::setw(5)<<nVarNodes<<"\n";
	std::cout<<std::setw(40)<<std::left<<"# Number of Edges Variables: "<<std::setw(5)<<nVarEdg<<"\n";
	std::cout<<std::setw(40)<<std::left<<"# Number of Nodes: "<<std::setw(5)<<nNodes<<"\n";
	std::cout<<std::setw(40)<<std::left<<"# Number of Hosts: "<<std::setw(5)<<nHosts<<"\n";
	std::cout<<std::setw(40)<<std::left<<"# Number of Edges: "<<std::setw(5)<<nEdges<<"\n";
	int* graph_groups=cluster->get_group_ptr();
	std::map<int,std::vector<int> > groups;
	for(int i=0; i<nNodes; i++) {
		groups[graph_groups[i]].push_back(i); // the group ID has the node i
	}
	std::cout<<"\tGroups formed "<<groups.size()<<"\n";
	for(auto it: groups) {
		std::cout<<"\t\tThe Group with ID "<<it.first<<" has "<<it.second.size()<<" nodes\n";
		for(auto el: it.second) {
			std::cout<<"\t\t\t"<<std::setw(20)<<nodeType[cluster->get_node_type(el)]<<" with ID "<<std::setw(4)<<el<<" witch containes: "<<cluster->get_variable_node(2,el)<<" as bandwidth\n";
		}
	}
	std::cout<<"\n\n";
}
