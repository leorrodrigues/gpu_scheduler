#ifndef _BEST_FIT_NOT_INCLUDED_
#define _BEST_FIT_NOT_INCLUDED_

#include <queue>

#include "../free.hpp"
#include "../utils.hpp"

struct CompareBF {
	bool operator()(Host* lhs, Host* rhs) const {
		std::map<std::string,float> l_r = lhs->getResource();
		std::map<std::string,float> r_r = rhs->getResource();
		float r1=0, r2=0;
		for(std::map<std::string,float>::iterator it_1 = l_r.begin(), it_2 = r_r.begin(); it_1!=l_r.end(); it_1++, it_2++) {
			r1+=it_1->second;
			r2+=it_2->second;
		}
		if(r1<r2) return true;
		else return false;
	}
};

namespace Allocator {
bool bestFit(Builder* builder,  Task* task, consumed_resource_t* consumed){
	std::vector<Host*> aux = builder->getHosts();
	std::priority_queue<Host*, std::vector<Host*>, CompareBF> hosts;


	Host* host=NULL;
	// printf("Get Task Pods\n");
	Pod** pods = task->getPods();
	unsigned int pods_size = task->getPodsSize();
	bool pod_allocated;

	for(size_t pod_index=0; pod_index < pods_size; pod_index++) {
		hosts = std::priority_queue<Host*, std::vector<Host*>, CompareBF>  (aux.begin(), aux.end());

		pod_allocated = false;
		host=NULL;

		while(!hosts.empty()) {
			host =  hosts.top();
			hosts.pop();

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

		hosts = std::priority_queue<Host*, std::vector<Host*>, CompareBF> ();

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
