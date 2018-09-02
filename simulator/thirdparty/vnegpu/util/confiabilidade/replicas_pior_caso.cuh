#ifndef _REPLICAS_PIOR_CASO_CUH
#define _REPLICAS_PIOR_CASO_CUH
#include <iostream>
#include <vnegpu/graph.cuh>
#include <vnegpu/util/confiabilidade/funcao_beta.cuh>

#define IND_VAR_PROBABILIDADE_FALHA 4

namespace vnegpu{

    namespace util{
        template <typename T, class VariablesType>
        int gerarReplicasPiorCaso(graph<T, VariablesType> *provedores, int qtd_vms_criticas, float confiabilidade){
            return qtd_vms_criticas*3;
            // float maior_probabilidade_falha = -1;
            //
            // for (int i = 0; i < provedores->get_num_nodes(); i++) {
            //     if(provedores->get_variable_node(IND_VAR_PROBABILIDADE_FALHA, i) > maior_probabilidade_falha){
            //         maior_probabilidade_falha = provedores->get_variable_node(IND_VAR_PROBABILIDADE_FALHA, i);
            //         // acha a zona com maior probabilidade de falha
            //     }
            // }
            // // chama a funcao beta passando a quantidade de vms criticas, a maior probabilidade de falha e a confiabilidade objetivo
            // return vnegpu::util::funcao_beta(qtd_vms_criticas, maior_probabilidade_falha, confiabilidade);
        }

    }
}

#endif
