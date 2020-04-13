#ifndef _AGRUPAMENTO_CUSTO_PROBABILIDADE_CUH
#define _AGRUPAMENTO_CUSTO_PROBABILIDADE_CUH

#include <iostream>
#include <utility>
#include <vnegpu/graph.cuh>
#include <vnegpu/util/confiabilidade/funcao_beta.cuh>

#define IND_VAR_TIPO_VM 1
#define IND_VAR_PROBABILIDADE_FALHA 4
#define IND_VAR_CUSTO_VM 5
#define IND_VAR_DST_PROBABILIDADE 1

#define CRITICA 1
#define REPLICA_ATIVADA 2
#define REPLICA_DESATIVADA 3

namespace vnegpu{

    namespace util{

        template <typename T, class VariablesType>
        void agruparCustoProbabilidade(graph<T, VariablesType> *provedores, int qtd_grupos_probabilidade, int qtd_grupos_custo){

            int ind_var_grupo_custo = provedores->add_node_variable("grupoCustoLink");

            provedores->update_gpu();

            vnegpu::algorithm::r_kleene(provedores, provedores->variables.edge_capacity);

            vnegpu::algorithm::k_means(provedores, qtd_grupos_custo, vnegpu::distance::matrix_based());

            provedores->update_cpu();

            for (int i = 0; i < provedores->get_num_nodes(); i++) {
                provedores->set_variable_node(ind_var_grupo_custo, i, provedores->get_group_id(i));
            }

            provedores->update_gpu();

            vnegpu::algorithm::r_kleene(provedores, IND_VAR_DST_PROBABILIDADE);

            vnegpu::algorithm::k_means(provedores, qtd_grupos_probabilidade, vnegpu::distance::matrix_based());

            provedores->update_cpu();


        }
    }
}

#endif
