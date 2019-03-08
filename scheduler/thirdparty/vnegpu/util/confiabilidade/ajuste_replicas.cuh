#ifndef _CALCULO_CONFIABILIDADE_CUH
#define _CALCULO_CONFIABILIDADE_CUH

#include <iostream>
#include <utility>
#include <vnegpu/graph.cuh>
#include <vnegpu/util/confiabilidade/funcao_beta.cuh>

#define IND_VAR_TIPO_VM 1
#define IND_VAR_PROBABILIDADE_FALHA 4
#define IND_VAR_CUSTO_VM 5

#define CRITICA 1
#define REPLICA_ATIVADA 2
#define REPLICA_DESATIVADA 3

namespace vnegpu{

    namespace util{

        int sum_vm_alocada(std::vector<int> vm_alocada){
            int total = 0;
            for (size_t i = 0; i < vm_alocada.size(); i++) {
                total += vm_alocada[i];
            }
            return total;
        }

        bool comparacao (std::pair<int, float> i,std::pair<int, float> j) { return (i.second < j.second); }

        template <typename T, class VariablesType>
        std::vector<int> ajustar_replicas(graph<T, VariablesType> *provedores, graph<T, VariablesType> *requisicao, float confiabilidade_objetivo){

            int replicas_necessarias = 0;

            std::vector<std::pair<int, float>> replicas_ativadas;
            std::vector<std::pair<int, float>> replicas_desativadas;
            std::vector<int> vm_alocada(provedores->get_num_nodes(), 0);
            for (int i = 0; i < requisicao->get_num_nodes(); i++) {
                int no_alocado = requisicao->get_allocation_to_nodes_ids(i);
                // std::cout << i << " - " << requisicao->get_allocation_to_nodes_ids(i) << " = " << provedores->get_variable_node(IND_VAR_PROBABILIDADE_FALHA, no_alocado) << std::endl;
                if(requisicao->get_variable_node(IND_VAR_TIPO_VM, i) < REPLICA_DESATIVADA) vm_alocada[no_alocado]++;
                if(requisicao->get_variable_node(IND_VAR_TIPO_VM, i) == CRITICA){
                    replicas_necessarias += funcao_beta(1, provedores->get_variable_node(IND_VAR_PROBABILIDADE_FALHA, no_alocado), confiabilidade_objetivo);
                }else if(requisicao->get_variable_node(IND_VAR_TIPO_VM, i) == REPLICA_ATIVADA){
                    replicas_ativadas.push_back(std::make_pair(i, provedores->get_variable_node(IND_VAR_CUSTO_VM, no_alocado)));
                }else if(requisicao->get_variable_node(IND_VAR_TIPO_VM, i) == REPLICA_DESATIVADA){
                    replicas_desativadas.push_back(std::make_pair(i, provedores->get_variable_node(IND_VAR_CUSTO_VM, no_alocado)));
                }

            }

            int total_replicas = replicas_ativadas.size() + replicas_desativadas.size();
            std::cout << sum_vm_alocada(vm_alocada) << std::endl;
            // std::cout << replicas_necessarias << std::endl;
            // std::cout << replicas_ativadas.size() << std::endl;

            std::sort(replicas_ativadas.begin(), replicas_ativadas.end(), comparacao);
            std::sort(replicas_desativadas.begin(), replicas_desativadas.end(), comparacao);

            // for (size_t i = 0; i < replicas_ativadas.size(); i++) {
            //     std::cout << replicas_ativadas[i].second << std::endl;
            // }


            while (replicas_desativadas.size() > total_replicas - replicas_necessarias) {
                requisicao->set_variable_node(IND_VAR_TIPO_VM, replicas_desativadas[0].first, REPLICA_ATIVADA);
                vm_alocada[requisicao->get_allocation_to_nodes_ids(replicas_desativadas[0].first)]++;
                replicas_desativadas.erase(replicas_desativadas.begin());
            }

            while(replicas_ativadas.size() > replicas_necessarias) {
                requisicao->set_variable_node(IND_VAR_TIPO_VM, replicas_ativadas[replicas_ativadas.size() - 1].first, REPLICA_DESATIVADA);
                vm_alocada[requisicao->get_allocation_to_nodes_ids(replicas_ativadas[replicas_ativadas.size() - 1].first)]--;
                replicas_ativadas.pop_back();
            }
            // std::cout << replicas_ativadas.size() << std::endl;
            // std::cout << replicas_necessarias << std::endl;

            return vm_alocada;
        }

    }
}

#endif
