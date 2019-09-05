#ifndef _BUILDER_NOT_INCLUDED_
#define _BUILDER_NOT_INCLUDED_

#include "types.hpp"

#include "multicriteria/ahp/ahp.hpp"
#include "multicriteria/ahp/ahpg.cuh"
#include "multicriteria/topsis/topsis.cuh"

#include "clustering/mclInterface.cuh"

#include "datacenter/tasks/task.hpp"
#include "datacenter/tasks/pod.hpp"
#include "datacenter/tasks/container.hpp"

#include <cmath>

class Builder : public main_resource_t {
private:
Multicriteria* multicriteriaMethod;
Multicriteria* multicriteriaClusteredMethod;
Clustering* clusteringMethod;
Topology* topology;
std::vector<Host*> hosts;
std::vector<Host*> clusterHosts;

void setMulticriteria(Multicriteria*);
void setClustering(Clustering*);
void setTopology(Topology*);

void parserHosts(const rapidjson::Value &);
void parserTopology(const rapidjson::Value &);
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
std::map<std::string,Interval_Tree::Interval_Tree*> getResource();
std::vector<Host*> getHosts();
Host* getHost(unsigned int);
size_t getHostsSize();
std::vector<Host*> getClusterHosts();
int getHostsMedianInGroup();
std::vector<Host*> getHostsInGroup(unsigned int);
bool findHostInGroup(unsigned int, unsigned int);
int getTotalActiveHosts();


void printClusterResult(int low, int high);
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
	const char* hostsSchemaPath = "datacenter/json/hostsSchema.json"
	);

};

#endif
