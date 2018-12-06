#ifndef _BUILDER_NOT_INCLUDED_
#define _BUILDER_NOT_INCLUDED_

#include "types.hpp"

#include "multicriteria/ahp.hpp"
#include "multicriteria/ahpg.cuh"

#include "clustering/mclInterface.cuh"

#include "rabbit/comunicator.hpp"

#include "datacenter/tasks/container.hpp"

#include <cmath>

class Builder {
private:
Multicriteria* multicriteriaMethod;
Multicriteria* multicriteriaClusteredMethod;
Clustering* clusteringMethod;
Topology* topology;
Resource resource;
std::vector<Host*> hosts;
std::vector<Host*> clusterHosts;
std::map<int,const char*> clusteredMulticriteria;

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
Multicriteria* getMulticriteriaClustered();
Clustering* getClustering();
Topology* getTopology();
std::map<int,const char*> getMulticriteriaResult();
std::map<int,const char*> getMulticriteriaClusteredResult();
int getClusteringResultSize();
void getClusteringResult();
Resource* getResource();
std::vector<Host*> getHosts();
Host* getHost(std::string name);
std::vector<Host*> getClusterHosts();
int getHostsMedianInGroup();
std::vector<Host*> getHostsInGroup(int);
int getTotalActiveHosts();


void printClusterResult();
void printTopologyType();
//Multicriteria set functions
void setAHP();
void setAHPG();
void setClusteredAHP();
void setClusteredAHPG();


//Clustering set functions
void setMCL();

//Topology set functions
void setFatTree(int);
void setBcube(int,int);
void setDcell(int,int);

// Set All Resources Data Center
void setDataCenterResources(total_resources_t*);
//Run functions methods
void runMulticriteria(std::vector<Host*> alt={});
void runMulticriteriaClustered(std::vector<Host*> alt={});
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
