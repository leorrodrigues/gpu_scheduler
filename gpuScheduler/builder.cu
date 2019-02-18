#include  "builder.cuh"
#include <assert.h>

Builder::Builder() : main_resource_t() {
	this->multicriteriaMethod=NULL;
	this->multicriteriaClusteredMethod=NULL;
	this->clusteringMethod=NULL;
	this->topology=NULL;
}

Builder::~Builder(){
	// printf("1\n");
	if(this->multicriteriaMethod)
		delete(this->multicriteriaMethod);
	// printf("2\n");
	if(this->multicriteriaClusteredMethod!=NULL)
		delete(this->multicriteriaClusteredMethod);
	// printf("3\n");
	if(this->clusteringMethod!=NULL)
		delete(this->clusteringMethod);
	// printf("4\n");
	if(this->topology!=NULL)
		delete(this->topology);
	// printf("5\n");
	this->multicriteriaMethod=NULL;
	// printf("6\n");
	this->multicriteriaClusteredMethod=NULL;
	// printf("7\n");
	this->clusteringMethod=NULL;
	// printf("8\n");
	this->topology=NULL;
	// printf("9\n");
	this->resource.clear();
	for(std::vector<Host*>::iterator it=hosts.begin(); it!=hosts.end(); it++) {
		delete(*it);
		*it=NULL;
	}
	// printf("11\n");
	hosts.clear();
	// printf("12\n");

	for(std::vector<Host*>::iterator it=clusterHosts.begin(); it!=clusterHosts.end(); it++) {
		delete(*it);
	}
	// printf("13\n");

	clusterHosts.clear();
	// printf("14\n");

}

Multicriteria* Builder::getMulticriteria(){
	return this->multicriteriaMethod;
}

Multicriteria* Builder::getMulticriteriaClustered(){
	return this->multicriteriaClusteredMethod;
}

Clustering* Builder::getClustering(){
	return this->clusteringMethod;
}

Topology* Builder::getTopology(){
	return this->topology;
}

unsigned int* Builder::getMulticriteriaResult(unsigned int& size){
	return this->multicriteriaMethod->getResult(size);
}

unsigned int* Builder::getMulticriteriaClusteredResult(unsigned int& size){
	return this->multicriteriaClusteredMethod->getResult(size);
}

int Builder::getClusteringResultSize(){
	return this->clusteringMethod->getResultSize();
}

void Builder::getClusteringResult(){
	//std::map<int,Host*> groups= this->clusteringMethod->getResult(this->topology,this->hosts);
	this->clusterHosts =  this->clusteringMethod->getResult(this->topology,this->hosts);
	//have to set the hosts group in the Clustering vector to use in Multicriteria.

}

std::map<std::string, float> Builder::getResource(){
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

void Builder::printClusterResult(){
	for(Host* it: this->clusterHosts) {
		std::cout<<it->getId()<<"\n";
		std::map<std::string, float> r=it->getResource();
		for(auto a: r) {
			std::cout<<"\t"<<a.first<<" "<<a.second<<"\n";
		}
	}
}

void Builder::printTopologyType(){
	std::cout<<this->topology->getTopology();
}

void Builder::setMulticriteria(Multicriteria* method){
	this->multicriteriaMethod=method;
}

void Builder::setAHP(){
	AHP *ahp=new AHP();
	this->setMulticriteria(ahp);
}

void Builder::setAHPG(){
	AHPG *ahpg=new AHPG();
	this->setMulticriteria(ahpg);
}

void Builder::setTOPSIS(){
	TOPSIS *topsis = new TOPSIS();
	this->setMulticriteria(topsis);
}

void Builder::setClusteredAHP(){
	AHP *ahp=new AHP();
	this->multicriteriaClusteredMethod=ahp;
}

void Builder::setClusteredAHPG(){
	AHPG *ahpg=new AHPG();
	this->multicriteriaClusteredMethod=ahpg;
}

void Builder::setClusteredTOPSIS(){
	TOPSIS *topsis= new TOPSIS();
	this->multicriteriaClusteredMethod=topsis;
}

void Builder::setClustering(Clustering* method){
	this->clusteringMethod=method;
}

void Builder::setMCL(){
	MCLInterface *mcl= new MCLInterface();
	this->setClustering(mcl);
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
	graph->listTopology();
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
	for(size_t i=0; i<this->hosts.size(); i++) {
		std::map<std::string,float> h_r = this->hosts[i]->getResource();
		for(auto it = resource->resource.begin(); it!=resource->resource.end(); it++) {
			resource->resource[it->first] += h_r[it->first];
		}
	}
}

void Builder::runMulticriteria(std::vector<Host*> alt){
	if(this->multicriteriaMethod!=NULL)
		this->multicriteriaMethod->run(&alt[0], alt.size());
}

void Builder::runMulticriteriaClustered(std::vector<Host*> alt){
	if(this->multicriteriaClusteredMethod!=NULL) {
		this->multicriteriaClusteredMethod->run(&alt[0], alt.size());
	}
}

void Builder::runClustering(std::vector<Host*> alt){
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
	std::map<std::string,float> resource = main_resource_t ().resource;
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
