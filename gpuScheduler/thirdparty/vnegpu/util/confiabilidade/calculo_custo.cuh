#ifndef _CALCULO_CUSTO_CUH
#define _CALCULO_CUSTO_CUH

#include <iostream>
#include <vnegpu/graph.cuh>

#define IND_VAR_TIPO_VM 1
#define IND_VAR_CUSTO_VM 5

#define REPLICA_ATIVADA 2
#define REPLICA_DESATIVADA 3


namespace vnegpu{

    namespace util{
        template <typename T, class VariablesType>
        int calcularCusto(graph<T, VariablesType> *provedores, graph<T, VariablesType> *requisicao){
            float custo_alocacao = 0;
            float custo_links = 0;

            for (int i = 0; i < requisicao->get_num_nodes(); i++) {
                if(requisicao->get_variable_node(IND_VAR_TIPO_VM, i) < REPLICA_DESATIVADA){
                    custo_alocacao += provedores->get_variable_node(IND_VAR_CUSTO_VM, requisicao->get_allocation_to_nodes_ids(i));
                }
            }
            // int total_links = 0;
            for (int i = 0; i < requisicao->get_num_edges(); i++) {
                int origem = 0, destino = 0, aloc_origem = 0, aloc_destino = 0;
                origem = i/requisicao->get_num_nodes();
                destino = i%requisicao->get_num_nodes();
                if(origem != destino && origem < destino){
                    if(requisicao->get_variable_node(IND_VAR_TIPO_VM, origem) < REPLICA_DESATIVADA && requisicao->get_variable_node(IND_VAR_TIPO_VM, destino) < REPLICA_DESATIVADA){
                        aloc_origem = requisicao->get_allocation_to_nodes_ids(origem);
                        aloc_destino = requisicao->get_allocation_to_nodes_ids(destino);
                        if(aloc_origem != aloc_destino){
                            // total_links++;
                            // std::cout << origem << " " << aloc_origem << " " << destino << " " << aloc_destino << std::endl;
                            // std::cout << provedores->get_variable_edge(provedores->variables.edge_capacity, aloc_origem*(provedores->get_num_nodes() - 1) + aloc_destino) << std::endl;
                            custo_links += provedores->get_variable_edge(provedores->variables.edge_capacity, aloc_origem*(provedores->get_num_nodes() - 1) + aloc_destino) * 500;
                        }
                    }
                }
            }
            // std::cout << total_links << std::endl;
            // std::cout << custo_alocacao << " " << custo_links << std::endl;
            return custo_alocacao + custo_links;
        }

    }
}

#endif
