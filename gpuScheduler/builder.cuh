#ifndef _BUILDER_NOT_INCLUDED_
#define _BUILDER_NOT_INCLUDED_

#include "multicriteria/ahp.hpp"
#include "multicriteria/ahpg.cuh"

#include "clustering/mclInterface.cuh"

#include "rabbit/comunicator.hpp"

#include "datacenter/tasks/container.hpp"

#include <cmath>

class Builder {
private:
Multicriteria* multicriteriaMethod;
Clustering* clusteringMethod;
Topology* topology;
Resource resource;
std::vector<Host*> hosts;
std::vector<Host*> clusterHosts;

void setMulticriteria(Multicriteria*);
void setClustering(Clustering*);
void setTopology(Topology*);

void generateContentSchema();

void addResource(std::string,std::string);
Host* addHost();

void parserDOM(JSON::jsonGenericDocument*);
void parserResources(JSON::jsonGenericType*);
void parserHosts(JSON::jsonGenericType*);
void parserTopology(JSON::jsonGenericType*);
public:

Builder();

Multicriteria* getMulticriteria();
Clustering* getClustering();
Topology* getTopology();
std::map<int,std::string> getMulticriteriaResult();
void getClusteringResult();
Resource* getResource();
std::vector<Host*> getHosts();
Host* getHost(std::string name);
std::vector<Host*> getClusterHosts();

void printTopologyType();
//Multicriteria set functions
void setAHP();
void setAHPG();

//Clustering set functions
void setMCL();

//Topology set functions
void setFatTree(int);
void setBcube(int,int);
void setDcell(int,int);

//Run functions methods
void runMulticriteria(std::vector<Host*> alt={});
void runClustering(std::vector<Host*> alt);

void listHosts();
void listResources();
void listCluster();
void parser(
	const char* hostsDataPath = "datacenter/json/fat_tree/20.json",
	const char* resourceDataPath = "datacenter/json/resourcesData.json",
	const char* hostsSchemaPath = "datacenter/json/hostsSchema.json",
	const char* resourceSchemaPath = "datacenter/json/resourcesSchema.json"
	);

};

#endif
