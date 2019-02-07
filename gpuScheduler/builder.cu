#include  "builder.cuh"
#include <assert.h>

Builder::Builder(){
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

void Builder::generateContentSchema() {
	std::string names;
	std::string text = "{\"$schema\":\"http://json-schema.org/draft-04/schema#\",\"definitions\":{\"topology\": {\"type\": \"object\",\"minProperties\": 1,\"additionalProperties\": false,\"properties\": {\"type\": {\"type\": \"string\"},\"size\": {\"type\": \"number\"},\"level\": {\"type\": \"number\"}},\"required\": [\"type\",\"size\"]},     \"host\": {\"type\": \"array\",\"minItems\": 1,\"items\":{\"properties\": {";
	for (auto it : this->resource) {
		//need to correct the type of the variable
		text += "\"" + it.first + "\":{\"type\":\"number\"},";
		names += "\"" + it.first + "\",";
	}
	names.pop_back();
	text.pop_back();
	text += "},\"additionalProperties\": false,\"required\": [" + names +        "]}}},\"type\": \"object\",\"minProperties\": 1,\"additionalProperties\": false,\"properties\": {\"topology\": {\"$ref\": \"#/definitions/topology\"}, \"hosts\": {\"$ref\": \"#/definitions/host\"}},\"required\": [\"topology\",\"hosts\"]}";
	JSON::writeJson("datacenter/json/hostsSchema.json", text);
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

std::vector<Host*> Builder::getHosts(){
	return this->hosts;
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

void Builder::addResource(std::string name){
	if(name!="id" && name!="name")
		this->resource[name] = 0;
}

Host* Builder::addHost() {
	//Call the host constructor (i.e., new host).
	Host* host = new Host(this->resource);
	//Add the host pointer in the hierarchy (i.e., the hosts vector).
	this->hosts.push_back(host);
	assert(host!=NULL);
	return host;
}

void Builder::setMulticriteria(Multicriteria* method){
	this->multicriteriaMethod=method;
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
	int i;
	int size=hosts.size();
	resource->servers = size;
	for(i=0; i < size; i++) {
		resource->vcpu += hosts[i]->getResource()["vcpu"];
		resource->ram += hosts[i]->getResource()["memory"];
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

void Builder::parserResources(JSON::jsonGenericType* dataResource) {
	std::string variableName, variableType;
	for (auto &arrayData : dataResource->value.GetArray()) {
		variableName = variableType = "";
		for (auto &objectData : arrayData.GetObject()) {
			if (strcmp(objectData.name.GetString(), "name") == 0) {
				variableName = objectData.value.GetString();
			} else if (strcmp(objectData.name.GetString(), "variableType") == 0) {
				variableType = strLower(objectData.value.GetString());
			} else {
				std::cout << "(builder.cu 321) Error in reading resources\nExiting...\n";
				exit(0);
			}
		}
		this->addResource(variableName);
		this->addResource("allocated_resources");
	}
}

void Builder::parserTopology(JSON::jsonGenericType* dataTopology){
	std::string topologyType;
	int size,level;
	for(auto &topology : dataTopology->value.GetObject()) {
		if(strcmp(topology.name.GetString(),"type")==0) {
			topologyType=topology.value.GetString();
		}
		else if(strcmp(topology.name.GetString(),"size")==0) {
			size=topology.value.GetInt();
		}
		else if(strcmp(topology.name.GetString(),"level")==0) {
			level=topology.value.GetInt();
		}
	}
	if(topologyType == "fattree" || topologyType == "fat tree" || topologyType == "fat_tree") {
		this->setFatTree(size);
	}else if(topologyType == "bcube") {
		this->setBcube(size,level);
	}else if(topologyType == "dcell") {
		this->setDcell(size,level);
	}else{
		std::cout<<"(builder.cu 350) unknow topology...\nexiting....\n";
		exit(1);
	}
}

void Builder::parserHosts(JSON::jsonGenericType* dataHost) {
	for (auto &arrayHost : dataHost->value.GetArray()) {
		Host* host = this->addHost();
		for (auto &alt : arrayHost.GetObject()) {
			std::string name(alt.name.GetString());
			if (alt.value.IsNumber()) {
				if(name!="id" && name!="name") {
					host->setResource(name, alt.value.GetFloat());
				}else{
					host->setId(alt.value.GetInt());
				}
			} else if (alt.value.IsBool()) {
				host->setResource(name, alt.value.GetBool());
			} else {
				printf("builder.cu(333) ERROR TYPE\n");
				exit(0);
			}
		}
	}
}

void Builder::parserDOM(JSON::jsonGenericDocument* data) {
	for (auto &m : data->GetObject()) { // query through all objects in data.
		if (strcmp(m.name.GetString(), "resources") == 0) {
			this->parserResources(&m);
		} else if (strcmp(m.name.GetString(), "hosts") == 0) {
			this->parserHosts(&m);
		} else if (strcmp(m.name.GetString(),"topology")==0) {
			this->parserTopology(&m);
		}
	}
}

void Builder::parser(
	const char* hostsDataPath,
	const char* resourceDataPath,
	const char* hostsSchemaPath,
	const char* resourceSchemaPath
	){
	//Parser the resources
	rapidjson::SchemaDocument resourcesSchema =
		JSON::generateSchema(resourceSchemaPath);
	rapidjson::Document resourcesData =
		JSON::generateDocument(resourceDataPath);
	rapidjson::SchemaValidator resourcesValidator(resourcesSchema);
	if (!resourcesData.Accept(resourcesValidator))
		JSON::jsonError(&resourcesValidator);
	parserDOM(&resourcesData);
	// generateContentSchema();
	//Parser the hosts
	rapidjson::SchemaDocument hostsSchema =
		JSON::generateSchema(hostsSchemaPath);
	rapidjson::Document hostsData =
		JSON::generateDocument(hostsDataPath);
	rapidjson::SchemaValidator hostsValidator(hostsSchema);
	if (!hostsData.Accept(hostsValidator))
		JSON::jsonError(&hostsValidator);
	parserDOM(&hostsData);
}
