#ifndef _FREE_ALLOCATION_
#define _FREE_ALLOCATION_

#include "../datacenter/tasks/task.hpp"
#include "../datacenter/tasks/pod.hpp"
#include "../datacenter/host.hpp"
#include "utils.hpp"

namespace Allocator {

inline void  freeHostResource(Pod* pod, Builder* builder, int low, int high){
	Host* host=pod->getHost();
	//Update the host resources
	host->removePod(low, high, pod->getResources());
	//Need to find the group that has this host and update their resource
	std::vector<Host*> groups = builder->getClusterHosts();
	for(size_t i=0; i<groups.size(); i++) {
		if(builder->findHostInGroup(groups[i]->getId(),host->getId())) {
			groups[i]->removePod(low, high, pod->getResources());
			break;
		}
	}
}

inline void freeHostResource(Task* task, Builder* builder, int low, int high){
	Pod** pods = task->getPods();
	for(size_t i=0; i< task->getPodsSize(); i++)
		freeHostResource(pods[i], builder, low, high);
}

inline void freeLinks(Task* task, consumed_resource_t* consumed, Builder* builder, size_t link_index){
	vnegpu::graph<float>* graph = builder->getTopology()->getGraph();

	int   *destination = task->getLinkDestination();
	int   *path_edge   = task->getLinkPathEdge();
	float *values    = task->getLinkValues();
	int   *path        = task->getLinkPath();
	int   *init        = task->getLinkInit();

	int walk_index=0;

	//need to undo all the link's allocation
	for(size_t p = 0; p<link_index; p++) {
		walk_index = destination[p];
		if(walk_index == -1 ) {
			continue; // the link is between two containers that are in the same host
		}
		while(path[init[p]+walk_index]!=-1) {
			graph->set_variable_edge(
				1,                                 //to refer to the bandwidth
				path_edge[init[p]+walk_index],                                 //to refer to the specific edge
				(graph->get_variable_edge( 1, path_edge[init[p]+walk_index]) + values[p])
				);

			consumed->total_bandwidth_consumed -= values[p];

			graph->sub_connection_edge( path_edge [init[p]+walk_index] );

			walk_index = path[init[p]+walk_index];
		}
	}
	//need to free all the elements
	free(destination);
	free(path_edge);
	free(values);
	free(path);
	free(init);
	task->setLinkDestination(NULL);
	task->setLinkPathEdge(NULL);
	task->setLinkValues(NULL);
	task->setLinkPath(NULL);
	task->setLinkInit(NULL);

	consumed->active_links = graph->get_num_active_edges();
}

inline void freeAllResources(Task* task, consumed_resource_t* consumed, Builder* builder, int low, int high){
	freeHostResource(task, builder, low, high);
	freeLinks(task,consumed,builder,task->getLinksSize());
}

}
#endif
