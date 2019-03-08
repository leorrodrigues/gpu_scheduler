#ifndef _FUNCAO_BETA_INVERTIDA_CUH
#define _FUNCAO_BETA_INVERTIDA_CUH

#include <iostream>
#include <cmath>
#include <unistd.h>
#include <libs/boost_1_61_0/boost/math/special_functions/beta.hpp>


namespace vnegpu{

    namespace util{
        int funcao_beta(int qtd_vms_criticas, float probabilidade_falha, float confiabilidade_objetivo){
            //inicia passando a quantidade de vms criticas, a probabilidade de falha da zona e a confiabilidade objetivo
            int qtd_replicas = 0;
            float confiabilidade_atual = 1 - probabilidade_falha;
            //confiabilidade_atual comeca com a confiabilidade da zona, se essa já for maior qua a confiabilidade objetivo retorna 0
            //caso contrario essa variavel irá receber o resultado da funcao beta avaliada no momento
            while (confiabilidade_atual < confiabilidade_objetivo){
                qtd_replicas++;
                //sobe a quantidade de replicas e roda a funcao beta incompleta
                confiabilidade_atual = boost::math::ibeta(qtd_vms_criticas, qtd_replicas, 1 - probabilidade_falha);
            }
            //assim que a confiabilidade esperada é atingida retorna a quantidade de replicas necessárias para tal
            // std::cout << qtd_replicas << std::endl;
            // if(qtd_replicas > qtd_vms_criticas) return qtd_vms_criticas;
            // std::cout << qtd_replicas << std::endl;
            return qtd_replicas;
        }

    }
}

#endif
