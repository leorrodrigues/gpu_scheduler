#ifndef _BUILDER_NOT_INCLUDED_
#define _BUILDER_NOT_INCLUDED_

#include "main_resources/types.hpp"

#include "allocator/rank_algorithms/rank.hpp"

#include "clustering/mclInterface.cuh"

#include "datacenter/tasks/task.hpp"
#include "datacenter/tasks/pod.hpp"
#include "datacenter/tasks/container.hpp"

#include <cmath>

class Builder : public main_resource_t {
private:
Rank* rankMethod;
Rank* rankClusteredMethod;
Clustering* clusteringMethod;
Topology* topology;
std::vector<Host*> hosts;
std::vector<Host*> clusterHosts;

void parserHosts(const rapidjson::Value &);
void parserTopology(const rapidjson::Value &);
public:

Builder();
~Builder();

Rank* getRank();
Rank* getRankClustered();
Clustering* getClustering();
Topology* getTopology();
unsigned int* getRankResult(unsigned int&);
unsigned int* getRankClusteredResult(unsigned int&);
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

//Clustering set functions
void setRank(Rank*);
void setClusteredRank(Rank*);
void setClustering(Clustering*);
void setTopology(Topology*);

//Topology set functions
void setFatTree(int);
void setBcube(int,int);
void setDcell(int,int);

// Set All Resources Data Center
void setDataCenterResources(total_resources_t*);
//Run functions methods
void runRank(std::vector<Host*> alt={},int low = 0, int high = 0);
void runRankClustered(std::vector<Host*> alt={},int low = 0, int high = 0);
void runClustering();

void listHosts();
void listResources();
void listCluster();
void parser(
	const char* hostsDataPath = "datacenter/fat_tree/20.json",
	const char* hostsSchemaPath = "datacenter/hostsSchema.json"
	);

};

#endif
