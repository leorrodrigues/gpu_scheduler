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
	std::map<int,std::vector<Host*> > host_groups;
	std::map<int,int> convertion;
	std::vector<Host*> vGroups;
	int* graph_groups=cluster->get_group_ptr();
	int group_index=0;
	for(int i=0; i<cluster->get_hosts(); i++) {
		if ( convertion.find(graph_groups[i]) == convertion.end() ) {
			// std::cout << "Key not presented\n";
			// if ( groups.find(convertion[graph_groups[i]]) == groups.end() ) { // if the group key don't exists
			//Host *host=new Host(topology->getResource());
			//std::cout<<host->getResource()->mIntSize<<"\n";
			//groups[i]=host;
			// std::cout<<"GRAPH_GROUPS[i] "<< graph_groups[i]<<" AND I "<<i<<"\n";
			convertion[graph_groups[i]]=group_index++;
			groups[convertion[graph_groups[i]]]=new Host(topology->getResource());
			groups[convertion[graph_groups[i]]]->setResource("name",std::to_string(convertion[graph_groups[i]]));//or ID
		} // else{
		// std::cout << "KEY FOUND\n";
		// }
		// std::cout<<"CONVERTION "<<convertion[graph_groups[i]]<<" GRAPH_GROUP "<<graph_groups[i]<<"\n";
		(*groups[convertion[graph_groups[i]]])+=(*hosts[i]); // the group ID has the node i
		host_groups[convertion[graph_groups[i]]].push_back(hosts[i]);
	}
	//a vazao (edge) do grupo é o menor valor destre os    hosts do grupo.
	for(auto it: groups) {
		vGroups.push_back(it.second);
	}
	this->host_groups=host_groups;
	return vGroups;
}

int MCLInterface::getHostsMedianInGroup(){
	int total=0;
	for(std::map<int,std::vector<Host*> >::iterator it=this->host_groups.begin(); it!=this->host_groups.end(); it++) {
		total+=it->second.size();
		// std::cout<<"Total "<<total<<"\n";
	}
	return total/this->host_groups.size();
}

std::vector<Host*> MCLInterface::getHostsInGroup(int group_index){
	// std::cout << "SIZE OF HOSTS " << this->host_groups[group_index].size() << " WITH ID "<<group_index<<"\n";
	return this->host_groups[group_index];
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
