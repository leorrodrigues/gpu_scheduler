#ifndef _BEST_FIT_NOT_INCLUDED_
#define _BEST_FIT_NOT_INCLUDED_

#include <queue>

#include "../free.hpp"
#include "../utils.hpp"

// struct CompareBF {
//      bool operator()(Host* lhs, Host* rhs) const {
//              std::map<std::string,float> l_r = lhs->getResource();
//              std::map<std::string,float> r_r = rhs->getResource();
//              float r1=0, r2=0;
//              for(std::map<std::string,float>::iterator it_1 = l_r.begin(), it_2 = r_r.begin(); it_1!=l_r.end(); it_1++, it_2++) {
//                      r1+=it_1->second;
//                      r2+=it_2->second;
//              }
//              if(r1<r2) return true;
//              else return false;
//      }
// };

static bool host_bf_compare(Host *lhs, Host *rhs){
	std::map<std::string,float> l_r = lhs->getResource();
	std::map<std::string,float> r_r = rhs->getResource();
	float r1=0, r2=0;
	for(std::map<std::string,float>::iterator it_1 = l_r.begin(), it_2 = r_r.begin(); it_1!=l_r.end(); it_1++, it_2++) {
		r1+=it_1->second;
		r2+=it_2->second;
	}
	return (r2<r1) ? r2 : r1;
}

namespace Allocator {
bool bestFit(Builder* builder,  Task* task, consumed_resource_t* consumed){
	std::vector<Host*> aux;

	std::vector<Host*>::iterator host_iterator;
	Host* host;
	// printf("Get Task Pods\n");
	Pod** pods = task->getPods();
	unsigned int pods_size = task->getPodsSize();
	bool pod_allocated;

	for(size_t pod_index=0; pod_index < pods_size; pod_index++) {
		aux = builder->getHosts();

		pod_allocated = false;
		host=NULL;

		while(!aux.empty()) {
			host_iterator = std::min_element(aux.begin(), aux.end(),host_bf_compare);

			host =  (*host_iterator); //get the iterator element
			aux.erase(host_iterator); //remove the element from vector

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
