#ifndef _AGRUPAR_ZONAS_CUH
#define _AGRUPAR_ZONAS_CUH

#include <iostream>
#include <vector>
#include <algorithm>

#include <vnegpu/metrics.cuh>

#include <vnegpu/algorithm/local_metric.cuh>


namespace vnegpu{

    namespace util{
        template <typename T, class VariablesType>
        int agrupar_zonas(graph<T, VariablesType> *provedores){
            int ind_var_grupo = provedores->add_node_variable("Grupo");

            provedores->update_gpu();

            vnegpu::algorithm::local_metric(provedores, ind_var_grupo, vnegpu::metrics::probabilidade_falha());

            provedores->update_cpu();

            std::vector<float> probabilidades_falha;

            for (int i = 0; i < provedores->get_num_nodes(); i++) {
                if(probabilidades_falha.size() == 0){
                    probabilidades_falha.push_back(provedores->get_variable_node(ind_var_grupo, i));
                }else{
                    bool ja_adicionado = false;
                    for (int j = 0; j < probabilidades_falha.size(); j++) {
                        if(probabilidades_falha[j] == provedores->get_variable_node(ind_var_grupo, i)) ja_adicionado = true;
                    }
                    if(!ja_adicionado) probabilidades_falha.push_back(provedores->get_variable_node(ind_var_grupo, i));
                }
            }

            std::sort(probabilidades_falha.begin(), probabilidades_falha.end());

            for (int i = 0; i < provedores->get_num_nodes(); i++) {
                for (int j = 0; j < probabilidades_falha.size(); j++) {
                    if(provedores->get_variable_node(ind_var_grupo, i) == probabilidades_falha[j]) provedores->set_variable_node(ind_var_grupo, i, j);
                }
            }

            return ind_var_grupo;
        }

    }
}

#endif
