#include  "builder.cuh"

Builder::Builder(){
	this->resource.mIntSize = 0;
	this->resource.mWeightSize = 0;
	this->resource.mStringSize = 0;
	this->resource.mBoolSize = 0;
	this->multicriteriaMethod=NULL;
	this->clusteringMethod=NULL;
}

void Builder::generateContentSchema() {
	std::string names;
	std::string text = "{\"$schema\":\"http://json-schema.org/draft-04/schema#\",\"definitions\":{\"topology\": {\"type\": \"object\",\"minProperties\": 1,\"additionalProperties\": false,\"properties\": {\"type\": {\"type\": \"string\"},\"size\": {\"type\": \"number\"},\"level\": {\"type\": \"number\"}},\"required\": [\"type\",\"size\"]},     \"host\": {\"type\": \"array\",\"minItems\": 1,\"items\":{\"properties\": {";
	auto resource = this->getResource();
	for (auto it : resource->mInt) {
		text += "\"" + it.first + "\":{\"type\":\"number\"},";
		names += "\"" + it.first + "\",";
	}
	for (auto it : resource->mWeight) {
		text += "\"" + it.first + "\":{\"type\":\"number\"},";
		names += "\"" + it.first + "\",";
	}
	for (auto it : resource->mBool) {
		text += "\"" + it.first + "\":{\"type\":\"boolean\"},";
		names += "\"" + it.first + "\",";
	}
	for (auto it : resource->mString) {
		text += "\"" + it.first + "\":{\"type\":\"string\"},";
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

Clustering* Builder::getClustering(){
	return this->clusteringMethod;
}

Topology* Builder::getTopology(){
	return this->topology;
}

std::map<int, std::string> Builder::getMulticriteriaResult(){
	return this->multicriteriaMethod->getResult();
}

int Builder::getClusteringResultSize(){
	return this->clusteringMethod->getResult(this->topology,this->hosts).size();
}

void Builder::getClusteringResult(){
	//std::map<int,Host*> groups= this->clusteringMethod->getResult(this->topology,this->hosts);
	this->clusterHosts =  this->clusteringMethod->getResult(this->topology,this->hosts);
	//have to set the hosts group in the Clustering vector to use in Multicriteria.

}

Resource* Builder::getResource(){
	return &(this->resource);
}

Host* Builder::getHost(std::string name){
	// std::cout << "Looking for "<<name<<"\n";
	for(Host* h : this->hosts) {
		if(name == h->getName()) {
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

std::vector<Host*> Builder::getHostsInGroup(int group_index){
	return this->clusteringMethod->getHostsInGroup(group_index);
}

void Builder::addResource(std::string name, std::string type){
	if (type == "int") {
		//Check if the type is int.
		//Create new entry in the int map.
		this->resource.mInt[name] = 0;
		this->resource.mIntSize++;
	} else if (type == "double" || type == "float") {
		//Check if the type is float or float, the variable will be in the same map.
		//Create new entry in the WeightType map.
		this->resource.mWeight[name] = 0;
		this->resource.mWeightSize++;
	} else if (type == "string" || type == "char*" || type == "char[]" || type == "char") {
		//Check if the type is string or other derivative.
		//Create new entry in the std::string map.
		this->resource.mString[name] = "";
		this->resource.mStringSize++;
	} else if (type == "bool" || type == "boolean") {
		//Check if the type is bool or boolean.
		//Create the new entry in the bool map.
		this->resource.mBool[name] = false;
		this->resource.mBoolSize++;
	} else {
		//If the type is unknow the program exit.
		std::cout << "Builder -> Unrecognizable type\nExiting...\n";
		exit(0);
	}
}

Host* Builder::addHost() {
	//Call the host constructor (i.e., new host).
	Host* host = new Host(this->resource);
	//Add the host pointer in the hierarchy (i.e., the hosts vector).
	this->hosts.push_back(host);
	return host;
}

void Builder::setMulticriteria(Multicriteria* method){
	this->multicriteriaMethod=method;
}

void Builder::printClusterResult(){
	for(auto it: this->clusterHosts) {
		std::cout<<it->getName()<<"\n";
		Resource* r=it->getResource();
		for(auto a: r->mWeight) {
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
	graph->setResource(&(this->resource));
	graph->populateTopology(this->hosts);
	this->setTopology(graph);
}

void Builder::setBcube(int nHosts,int nLevels){
	Bcube* graph=new Bcube();
	graph->setSize(nHosts);
	graph->setLevel(nLevels);
	graph->setTopology();
	graph->setResource(&(this->resource));
	graph->populateTopology(this->hosts);
	this->setTopology(graph);
}

void Builder::setDcell(int nHosts,int nLevels){
	Dcell* graph=new Dcell();
	graph->setSize(nHosts);
	graph->setLevel(nLevels);
	graph->setTopology();
	graph->setResource(&(this->resource));
	graph->populateTopology(this->hosts);
	this->setTopology(graph);
}

void Builder::runMulticriteria(std::vector<Host*> alt){
	if(this->multicriteriaMethod!=NULL)
		this->multicriteriaMethod->run(alt);
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
		std::cout << "Host: "<<host->getName() <<"\n";
		std::cout<< "VCPU: "<<host->getResource()->mWeight["vcpu"]<<"\n";
		std::cout<< "RAM: "<<host->getResource()->mWeight["memory"]<<"\n";
	}
}

void Builder::listResources() {
	if (this->resource.mIntSize) {
		std::cout << "Int Resources\n";
		for (auto it : this->resource.mInt) {
			std::cout << "\t" << it.first << " : " << it.second << "\n";
		}
	}
	if (this->resource.mWeightSize) {
		std::cout << "Float/float Resources\n";
		for (auto it : this->resource.mWeight) {
			std::cout << "\t" << it.first << " : " << it.second << "\n";
		}
	}
	if (this->resource.mStringSize) {
		std::cout << "String Resources\n";
		for (auto it : this->resource.mString) {
			std::cout << "\t" << it.first << " : " << it.second << "\n";
		}
	}
	if (this->resource.mBoolSize) {
		std::cout << "Boolean Resources\n";
		for (auto it : this->resource.mBool) {
			std::cout << "\t" << it.first << " : " << it.second << "\n";
		}
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
				std::cout << "Error in reading resources\nExiting...\n";
				exit(0);
			}
		}
		this->addResource(variableName, variableType);
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
		std::cout<<"unknow topology...\nexiting....\n";
		exit(1);
	}
}

void Builder::parserHosts(JSON::jsonGenericType* dataHost) {
	for (auto &arrayHost : dataHost->value.GetArray()) {
		auto host = this->addHost();
		for (auto &alt : arrayHost.GetObject()) {
			std::string name(alt.name.GetString());
			if (alt.value.IsNumber()) {
				if (host->getResource()->mInt.count(name) > 0) {
					host->setResource(name, alt.value.GetInt());
				} else {
					host->setResource(name, alt.value.GetFloat());
				}
			} else if (alt.value.IsBool()) {
				host->setResource(name, alt.value.GetBool());
			} else {
				host->setResource(name, strLower(std::string(alt.value.GetString())));
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
	generateContentSchema();
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
