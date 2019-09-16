#include  "builder.cuh"
#include <assert.h>

Builder::Builder() : main_resource_t() {
	this->rankMethod=NULL;
	this->rankClusteredMethod=NULL;
	this->clusteringMethod=NULL;
	this->topology=NULL;
}

Builder::~Builder(){
	if(this->rankMethod != NULL)
		delete(this->rankMethod);
	if(this->clusteringMethod!=NULL)
		delete(this->clusteringMethod);
	if(this->topology!=NULL)
		delete(this->topology);
	this->rankMethod=NULL;
	this->rankClusteredMethod=NULL;
	this->clusteringMethod=NULL;
	this->topology=NULL;
	this->resource.clear();
	for(std::vector<Host*>::iterator it=hosts.begin(); it!=hosts.end(); it++) {
		delete(*it);
		*it=NULL;
	}
	hosts.clear();
	for(std::vector<Host*>::iterator it=clusterHosts.begin(); it!=clusterHosts.end(); it++) {
		delete(*it);
	}
	clusterHosts.clear();
}

Rank* Builder::getRank(){
	return this->rankMethod;
}

Rank* Builder::getRankClustered(){
	return this->rankClusteredMethod;
}

Clustering* Builder::getClustering(){
	return this->clusteringMethod;
}

Topology* Builder::getTopology(){
	return this->topology;
}

unsigned int* Builder::getRankResult(unsigned int& size){
	return this->rankMethod->getResult(size);
}

unsigned int* Builder::getRankClusteredResult(unsigned int& size){
	return this->rankClusteredMethod->getResult(size);
}

int Builder::getClusteringResultSize(){
	return this->clusteringMethod->getResultSize();
}

void Builder::getClusteringResult(){
	//std::map<int,Host*> groups= this->clusteringMethod->getResult(this->topology,this->hosts);
	this->clusterHosts =  this->clusteringMethod->getResult(this->topology,this->hosts);
	//have to set the hosts group in the Clustering vector to use in Multicriteria.

}

std::map<std::string, Interval_Tree::Interval_Tree*> Builder::getResource(){
	return this->resource;
}

std::vector<Host*> Builder::getHosts(){
	return this->hosts;
}


Host* Builder::getHost(unsigned int id){
	// std::cout << "Looking for "<<name<<"\n";
	for(Host* h : this->hosts) {
		if(id == h->getId()) {
			// std::cout<<" Name Found!\n";
			return h;
		}
	}
	// std::cout<<" Name Not Found!\n";
	return NULL;
}

size_t Builder::getHostsSize(){
	return this->hosts.size();
}

std::vector<Host*> Builder::getClusterHosts(){
	return this->clusterHosts;
}

int Builder::getHostsMedianInGroup(){
	return this->clusteringMethod->getHostsMedianInGroup();
}

std::vector<Host*> Builder::getHostsInGroup(unsigned int group_index){
	return this->clusteringMethod->getHostsInGroup(group_index);
}

bool Builder::findHostInGroup(unsigned int group_index, unsigned int host_id){
	return this->clusteringMethod->findHostInGroup(group_index, host_id);
}

int Builder::getTotalActiveHosts(){
	int total=0;
	int i, size = hosts.size();
	for(i=0; i<size; i++) {
		if(hosts[i]->getActive())
			total++;
	}
	return total;
}

void Builder::printClusterResult(int low, int high){
	for(Host* it: this->clusterHosts) {
		std::cout<<it->getId()<<"\n";
		std::map<std::string, Interval_Tree::Interval_Tree*> r=it->getResource();
		for(auto a: r) {
			std::cout<<"\t"<<a.first<<" "<<a.second->getMinValueAvailable(low, high)<<"\n";
		}
	}
}

void Builder::printTopologyType(){
	std::cout<<this->topology->getTopology();
}

void Builder::setRank(Rank* method){
	this->rankMethod=method;
}

void Builder::setClusteredRank(Rank* method){
	this->rankClusteredMethod=method;
}

void Builder::setClustering(Clustering* method){
	this->clusteringMethod=method;
}

void Builder::setTopology(Topology* topology){
	this->topology=topology;
}

void Builder::setFatTree(int k){
	FatTree* graph=new FatTree();
	graph->setSize(k);
	graph->setTopology();
	graph->setResource(this->resource);
	graph->populateTopology(this->hosts);
	this->setTopology(graph);
}

void Builder::setBcube(int nHosts,int nLevels){
	Bcube* graph=new Bcube();
	graph->setSize(nHosts);
	graph->setLevel(nLevels);
	graph->setTopology();
	graph->setResource(this->resource);
	graph->populateTopology(this->hosts);
	this->setTopology(graph);
}

void Builder::setDcell(int nHosts,int nLevels){
	Dcell* graph=new Dcell();
	graph->setSize(nHosts);
	graph->setLevel(nLevels);
	graph->setTopology();
	graph->setResource(this->resource);
	graph->populateTopology(this->hosts);
	this->setTopology(graph);
}

void Builder::setDataCenterResources(total_resources_t* resource){
	resource->servers = this->hosts.size();
	resource->links = this->topology->getGraph()->get_num_edges(); //to simulate the 1GB of each link
	std::map<std::string, Interval_Tree::Interval_Tree*> h_r;
	// spdlog::debug("Start iteration in data center resources\n");
	for(size_t i = 0; i < this->hosts.size(); i++) {
		h_r = this->hosts[i]->getResource();
		for(auto it = resource->resource.begin(); it!=resource->resource.end(); it++) {
			(*it->second) += h_r[it->first]->getCapacity();
		}
	}
	//
	// for(auto it = resource->resource.begin(); it!= resource->resource.end(); it++) {
	// 	spdlog::debug("Name {}",it->first);
	// 	it->second->show();
	// }
	//As the DC is configured to have 1GB in all links, only multiply by the links bandwidth by the total ammount
	resource->total_bandwidth = 1000*resource->links;
}

void Builder::runRank(std::vector<Host*> alt, int low, int high){
	if(this->rankMethod!=NULL)
		this->rankMethod->run(alt, alt.size(), low, high);
}

void Builder::runRankClustered(std::vector<Host*> alt, int low, int high){
	if(this->rankClusteredMethod!=NULL) {
		this->rankClusteredMethod->run(alt, alt.size(), low, high);
	}
}

void Builder::runClustering(){
	if(this->clusteringMethod!=NULL && this->topology!=NULL)
		this->clusteringMethod->run(this->topology);
}

/*List Functions*/

std::string strLower(std::string s) {
	std::transform(s.begin(), s.end(), s.begin(),[](unsigned char c){
		return std::tolower(c);
	});
	return s;
}

void Builder::listHosts(){
	for(Host* host: this->hosts) {
		std::cout << "Host: "<<host->getId() <<"\n";
		std::cout<< "VCPU: "<<host->getResource()["vcpu"]<<"\n";
		std::cout<< "RAM: "<<host->getResource()["memory"]<<"\n";
	}
}

void Builder::listResources() {
	std::cout << "Float/float Resources\n";
	for (auto it : this->resource) {
		std::cout << "\t" << it.first << " : " << it.second << "\n";
	}
}

void Builder::listCluster(){
	if(this->clusteringMethod!=NULL && this->topology!=NULL) {
		this->clusteringMethod->listGroups(this->topology);
	}
}

/*Parser Functions*/

void Builder::parserTopology(const rapidjson::Value &dataTopology){
	std::string topologyType (dataTopology["type"].GetString());
	unsigned int size = dataTopology["size"].GetInt();
	unsigned int level;
	if(dataTopology.HasMember("level"))
		level = dataTopology["level"].GetInt();

	if(topologyType == "fat_tree" || topologyType == "fat tree" || topologyType == "fattree") {
		this->setFatTree(size);
	}else if(topologyType == "bcube") {
		this->setBcube(size,level);
	}else if(topologyType == "dcell") {
		this->setDcell(size,level);
	}else{
		std::cout<<"(builder.cu) unknow topology...\nexiting....\n";
		exit(1);
	}
}

void Builder::parserHosts(const rapidjson::Value &dataHost) {
	Host* host=NULL;
	std::map<std::string,Interval_Tree::Interval_Tree*> resource = main_resource_t ().resource;
	for(size_t i=0; i<dataHost.Size(); i++) {
		host = new Host();
		this->hosts.push_back(host);
		host->setId(dataHost[i]["id"].GetInt());
		for(auto it = resource.begin(); it!=resource.end(); it++) {
			host->setResource(it->first,dataHost[i][it->first.c_str()].GetFloat());
		}
	}
}

void Builder::parser(const char* hostsDataPath,const char* hostsSchemaPath){
	//Parser the hosts and topology
	rapidjson::SchemaDocument hostsSchema =
		JSON::generateSchema(hostsSchemaPath);
	rapidjson::Document hostsData =
		JSON::generateDocument(hostsDataPath);
	rapidjson::SchemaValidator hostsValidator(hostsSchema);
	if (!hostsData.Accept(hostsValidator))
		JSON::jsonError(&hostsValidator);

	this->parserHosts(hostsData["hosts"]);
	this->parserTopology(hostsData["topology"]);
}
