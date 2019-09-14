#ifndef _BEST_FIT_NOT_INCLUDED_
#define _BEST_FIT_NOT_INCLUDED_

#include <queue>

#include "../free.hpp"
#include "../utils.hpp"

static size_t get_min_element(std::vector<Host*> hosts, bool *visited, int low, int high){
	float max = FLT_MAX;
	float temp = 0;
	size_t index=0;
	std::map<std::string,Interval_Tree::Interval_Tree*> l_r;
	for(size_t i=0; i<hosts.size(); i++) {
		// spdlog::info("\tAnalysing the {} host of {} hosts", i, hosts.size());
		if(!visited[i]) {
			temp=0;
			l_r = hosts[i]->getResource();
			temp += l_r["vcpu"]->getMinValueAvailable(low, high);
			temp += l_r["ram"]->getMinValueAvailable(low, high);
			if(max > temp) {
				max=temp;
				index=i;
			}
		}
	}
	return index;
}

namespace Allocator {
bool bestFit(Builder* builder,  Task* task, consumed_resource_t* consumed, int current_time){
	std::vector<Host*> aux = builder->getHosts();
	size_t hosts_size = aux.size();
	size_t visited_qnt=0;
	size_t host_index;
	Host* host;
	// printf("Get Task Pods\n");
	Pod** pods = task->getPods();
	unsigned int pods_size = task->getPodsSize();
	int interval_low = 0, interval_high = 0;
	size_t pod_index = 0;
	bool pod_allocated;
	bool fit = false;

	// spdlog::info("BF - INIT");
	for(interval_low = current_time; interval_low < task->getDeadline(); interval_low++) {
		interval_high = interval_low + task->getDuration();
		// spdlog::info("BF - Looking the interval [{},{}]", interval_low, interval_high);
		for(pod_index=0; pod_index < pods_size; pod_index++) {
			// spdlog::info("BF - Looking the pod {} of {}", pod_index+1, pods_size);
			bool visited [aux.size()];
			pod_allocated = false;
			host=NULL;
			while(visited_qnt<hosts_size) {
				// spdlog::info("BF - Looking the host {} in {} hosts", visited_qnt+1, hosts_size);
				fit = false;

				// spdlog::info("BF - Get the min host");
				host_index = get_min_element(aux,visited, interval_low, interval_high);

				host =  aux[host_index];//get the iterator element
				visited[host_index]=true; //remove the element from vector
				visited_qnt++;

				// spdlog::info("BF - check if fits");
				if(checkFit(host,pods[pod_index], interval_low, interval_high))
					fit = true;

				if(!fit) continue;

				// spdlog::info("A REQUISICAO FOI ACEITA NO HOST {}!\n", host->getId());
				std::map<std::string,std::vector<float> > p_r = pods[pod_index]->getResources();
				// spdlog::info("Adicionando o pod {} ao host {}", pod_index, host->getId());
				host->addPod(interval_low, interval_high, p_r);

				// spdlog::info("Verificando se o host esta ativo");
				if(!host->getActive()) {
					host->setActive(true);
					++consumed->active_servers;
				}
				// spdlog::info("Adicionando o consumo geral do DC");
				addToConsumed(consumed,pods[pod_index]);

				// spdlog::info("Configurando o id do host ao pod");
				pods[pod_index]->setHost(host);

				pod_allocated=true;
				break;
			}
			//need to desalocate all the allocated pods.
			if(!pod_allocated) {
				for(size_t i=0; i< pod_index; i++) {
					freeHostResource(pods[i],consumed,builder, interval_low, interval_high);
				}
				break;
			}
		}
		if(pod_allocated) {
			// spdlog::info("Todos os pods da task foram adicionados com sucesso\n");
			break;
		}
	}
	if(!pod_allocated)
		return false;
	// spdlog::info("Calculando o delay da task {}", interval_low - current_time);
	task->addDelay(interval_low - current_time);
	return true;
}


}
#endif
