#ifndef _BUILDER_NOT_INCLUDED_
#define _BUILDER_NOT_INCLUDED_

#include "types.hpp"

#include "multicriteria/ahp/ahp.hpp"
#include "multicriteria/ahp/ahpg.cuh"
#include "multicriteria/topsis/topsis.cuh"

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
std::map<std::string,float> resource;
std::vector<Host*> hosts;
std::vector<Host*> clusterHosts;

void setMulticriteria(Multicriteria*);
void setClustering(Clustering*);
void setTopology(Topology*);

void generateContentSchema();

void addResource(std::string);
Host* addHost();

void parserDOM(JSON::jsonGenericDocument*);
void parserResources(JSON::jsonGenericType*);
void parserHosts(JSON::jsonGenericType*);
void parserTopology(JSON::jsonGenericType*);
public:

Builder();
~Builder();

Multicriteria* getMulticriteria();
Multicriteria* getMulticriteriaClustered();
Clustering* getClustering();
Topology* getTopology();
unsigned int* getMulticriteriaResult(unsigned int&);
unsigned int* getMulticriteriaClusteredResult(unsigned int&);
int getClusteringResultSize();
void getClusteringResult();
std::map<std::string,float> getResource();
std::vector<Host*> getHosts();
Host* getHost(unsigned int);
std::vector<Host*> getClusterHosts();
int getHostsMedianInGroup();
std::vector<Host*> getHostsInGroup(unsigned int);
bool findHostInGroup(unsigned int, unsigned int);
int getTotalActiveHosts();


void printClusterResult();
void printTopologyType();
//Multicriteria set functions
void setAHP();
void setAHPG();
void setTOPSIS();
void setClusteredAHP();
void setClusteredAHPG();
void setClusteredTOPSIS();

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
