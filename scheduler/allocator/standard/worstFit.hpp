#ifndef _WORST_FIT_NOT_INCLUDED_
#define _WORST_FIT_NOT_INCLUDED_

#include <queue>

#include "../free.hpp"
#include "../utils.hpp"

static size_t get_max_element(std::vector<Host*> hosts, bool *visited, int low, int high){
	float max = FLT_MIN;
	float temp = 0;
	size_t index=0;
	std::map<std::string, Interval_Tree::Interval_Tree*> l_r;
	for(size_t i=0; i<hosts.size(); i++) {
		if(!visited[i]) {
			temp=0;
			l_r = hosts[i]->getResource();
			// for(std::map<std::string,float>::iterator it = l_r.begin(); it!=l_r.end(); it++) {
			// temp+=it->second;
			// }
			temp += l_r["vcpu"]->getMinValueAvailable(low, high);
			temp += l_r["ram"]->getMinValueAvailable(low, high);
			if(max < temp) {
				max=temp;
				index=i;
			}
		}
	}
	return index;
}

namespace Allocator {
bool worstFit(Builder* builder,  Task* task, consumed_resource_t* consumed, int interval_low){
	std::vector<Host*> aux = builder->getHosts();
	size_t hosts_size = aux.size();
	size_t visited_qnt=0;
	size_t host_index;
	Host* host;
	// printf("Get Task Pods\n");
	Pod** pods = task->getPods();
	unsigned int pods_size = task->getPodsSize();
	bool pod_allocated;
	int interval_high = 0;
	bool fit = false;

	for(size_t pod_index=0; pod_index < pods_size; pod_index++) {
		bool visited [aux.size()];
		pod_allocated = false;
		host=NULL;
		while(visited_qnt<hosts_size) {
			fit = false;
			host_index = get_max_element(aux,visited, interval_low, interval_high);

			host =  aux[host_index]; //get the iterator element
			visited[host_index]=true; //remove the element from vector
			visited_qnt++;

			for(interval_high = interval_low+1; interval_high < task->getDeadLine(); interval_high++) {
				if(checkFit(host,pods[pod_index], interval_low, interval_high)) {
					fit = true;
					break;
				}

			}
			if(!fit) continue;

			std::map<std::string,std::vector<float> > p_r = pods[pod_index]->getResources();
			host->addPod(interval_low, interval_high, p_r);

			if(!host->getActive()) {
				host->setActive(true);
				consumed->active_servers++;
			}

			addToConsumed(consumed,pods[pod_index], interval_low, interval_high);

			pods[pod_index]->setHost(host);

			pod_allocated=true;
			break;
		}

		if(!pod_allocated) {
			//need to desalocate all the allocated pods.
			for(size_t i=0; i< pod_index; i++)
				freeHostResource(pods[i],consumed,builder, interval_low, interval_high);
			return false;
		}

	}
	return true;
}


}
#endif
