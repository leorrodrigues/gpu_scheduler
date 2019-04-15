#ifndef _BEST_FIT_NOT_INCLUDED_
#define _BEST_FIT_NOT_INCLUDED_

#include <queue>

#include "../free.hpp"
#include "../utils.hpp"

static size_t get_min_element(std::vector<Host*> hosts, bool *visited){
    float max = FLT_MAX;
    float temp = 0;
    size_t index=0;
    std::map<std::string,float> l_r;
    for(size_t i=0; i<hosts.size();i++){
        if(!visited[i]){
            l_r = hosts[i]->getResource();
            for(std::map<std::string,float>::iterator it = l_r.begin(); it!=l_r.end(); it++) {
        		temp+=it->second;
        	}
            if(max > temp){
                max=temp;
                index=i;
            }
        }
    }
    return index;
}

namespace Allocator {
bool bestFit(Builder* builder,  Task* task, consumed_resource_t* consumed){
	std::vector<Host*> aux = builder->getHosts();

	size_t host_index;
	Host* host;
	// printf("Get Task Pods\n");
	Pod** pods = task->getPods();
	unsigned int pods_size = task->getPodsSize();
	bool pod_allocated;


	for(size_t pod_index=0; pod_index < pods_size; pod_index++) {
        bool visited [aux.size()];
		pod_allocated = false;
		host=NULL;
		while(!aux.empty()) {
			host_index = get_min_element(aux,visited);

			host =  aux[host_index]; //get the iterator element
			visited[host_index]=true; //remove the element from vector

			if(!checkFit(host,pods[pod_index])) continue;

			std::map<std::string,std::tuple<float,float,bool> > p_r = pods[pod_index]->getResources();
			host->addPod(p_r);

			if(!host->getActive()) {
				host->setActive(true);
				consumed->active_servers++;
			}

			addToConsumed(consumed,pods[pod_index]);

			pods[pod_index]->setHost(host);

			pod_allocated=true;
			break;
		}

		if(!pod_allocated) {
			//need to desalocate all the allocated pods.
			for(size_t i=0; i< pod_index; i++)
				freeHostResource(pods[i],consumed,builder);
			return false;
		}

	}
	return true;
}


}
#endif
