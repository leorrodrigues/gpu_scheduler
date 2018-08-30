#ifndef _GERAR_DADOS_CUH
#define _GERAR_DADOS_CUH

#include <iostream>
#include <map>
#include <utility>
#include <algorithm>
#include <vnegpu/graph.cuh>


#define IND_VAR_PROVEDOR 1
#define IND_VAR_REGIAO 2

#define IND_VAR_TIPO_VM 1

#define REPLICA_DESATIVADA 3

namespace vnegpu {

namespace util {

template <typename T, class VariablesType>
void gerarDados(graph<T, VariablesType> *provedores, graph<T, VariablesType> *requisicao, std::vector<int> vm_alocada){
	int qtd_vms_total = 0;
	int total_provedores = 0, total_regioes = 0, total_zonas = provedores->get_num_nodes();
	int regioes_por_provedor = 0;
	int regioes_usadas = 0, provedores_usados = 0, zonas_usadas = 0;
	bool provedor_ja_usado = false;
	bool regiao_ja_usada = false;
	for (int i = 0; i < provedores->get_num_nodes(); i++) {
		qtd_vms_total += vm_alocada[i];
		int provedor_atual = provedores->get_variable_node(IND_VAR_PROVEDOR, i);
		int regiao_atual = provedores->get_variable_node(IND_VAR_REGIAO, i);
		if(provedor_atual > total_provedores) {
			provedor_ja_usado = false;
			total_provedores = provedor_atual;
			total_regioes += regioes_por_provedor;
			regioes_por_provedor = 0;
		}
		if(regiao_atual > regioes_por_provedor) {
			regiao_ja_usada = false;
			regioes_por_provedor++;
		}
		if(provedores->get_variable_node(provedores->variables.node_capacity, i) > 0 && vm_alocada[i]) {
			if(!provedor_ja_usado) {
				provedor_ja_usado = true;
				provedores_usados++;
			}
			if(!regiao_ja_usada) {
				regiao_ja_usada = true;
				regioes_usadas++;
			}
			zonas_usadas++;
		}
	}
	int max_esp_provedores = std::min(total_provedores, qtd_vms_total);
	int max_esp_regioes = std::min(total_regioes, qtd_vms_total);
	int max_esp_zonas = std::min(total_zonas, qtd_vms_total);
	int max_esp_links = (max_esp_zonas * (max_esp_zonas - 1))/2;
	int links_vms_originais = max_esp_links;

	std::map<int, int> vms_ativas_zonas;
	for (int i = 0; i < requisicao->get_num_nodes(); i++) {
		if(requisicao->get_variable_node(IND_VAR_TIPO_VM, i) < REPLICA_ATIVADA) {
			int zona_alocada = requisicao->get_allocation_to_nodes_ids(i);
			vms_ativas_zonas[zona_alocada]++;
		}
	}

	for(std::map<int, int>::iterator it_map = vms_ativas_zonas.begin(); it_map != vms_ativas_zonas.end(); it_map++) {
		links_vms_originais -= (it_map->second * (it_map->second - 1))/2;
	}

	int links_replicas = max_esp_links - links_vms_originais;

	std::cout << max_esp_zonas << " " << zonas_usadas << " " << qtd_vms_total << std::endl;
	std::cout << max_esp_links << std::endl;
	std::cout << links_vms_originais << std::endl;
	// std::cout << max_esp_provedores << " " << provedores_usados << std::endl;
	// std::cout << max_esp_regioes << " " << regioes_usadas << std::endl;
	// std::cout << provedores_usados/max_esp_provedores << std::endl;
}
}
}

#endif
