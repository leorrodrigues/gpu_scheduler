#ifndef _REQUISICAO_CUH
#define _REQUISICAO_CUH

/*! \file
 *  \brief The generator of bcubes.
 */

// #include <iostream>
#include <fstream>
#include <string>
#include <vnegpu/graph.cuh>
#include <vnegpu/util/util.cuh>

#define REGULAR 0
#define CRITICA 1
#define REPLICA_ATIVADA 2
#define REPLICA_DESATIVADA 3

namespace vnegpu
{
  namespace generator
  {

    template <typename T, class VariablesType=vnegpu::graph_type::minimalist>
    vnegpu::graph<T,VariablesType>* requisicao(int qtd_vms, int qtd_vms_criticas, int qtd_replicas){
        // std::cout << qtd_replicas << std::endl;
        int qtd_total_vms = qtd_vms + qtd_vms_criticas + qtd_replicas;
        int qtdLinks = ((qtd_total_vms - 1) * qtd_total_vms)/2;
        vnegpu::graph<T,VariablesType>* requisicao = new vnegpu::graph<T,VariablesType>(qtd_total_vms, qtdLinks, VariablesType::total_nodes_variables, VariablesType::total_edges_variables);

        int ind_var_tipo_vm = requisicao->add_node_variable("Tipo");

        //percorre todos os nós estabelecendo os links através do vetor SO e DI (artigo)
        for (int i = 0; i < qtd_total_vms; i++) {
            // seta o source offset para cada nó
            requisicao->set_source_offsets(i, i*(qtd_total_vms - 1));
            requisicao->set_node_type(i, vnegpu::TYPE_HOST);
            if(i < qtd_vms_criticas){
                requisicao->set_variable_node(ind_var_tipo_vm, i, (float) CRITICA);
            }else if(i < qtd_vms_criticas + qtd_replicas){
                requisicao->set_variable_node(ind_var_tipo_vm, i, (float) REPLICA_ATIVADA);
            }else{
                requisicao->set_variable_node(ind_var_tipo_vm, i, (float) REGULAR);
            }
            // std::cout << qtd_vms + qtd_vms_criticas << std::endl;

            // preenche os destination indices do nó em questão
            for (int j = 0; j < qtd_total_vms; j++) {
                int correcao_autolink = 0;
                if(j > i) correcao_autolink = 1;
                if(j != i){
                     requisicao->set_destination_indices(i*(qtd_total_vms - 1) + j - correcao_autolink, j);
                }
            }
        }

        requisicao->set_source_offsets(qtd_total_vms, qtdLinks*2); //qtdlinks já tem a quantidade de arestas não dirigidas
        requisicao->check_edges_ids();
        requisicao->set_hosts(qtd_total_vms);

        //mostra o vetor de SO e o DI:
        // std::cout << std::endl << "quantidade de nós: " << requisicao->get_num_nodes() << std::endl << "source offsets: ";
        // for (int i = 0; i < requisicao->get_num_nodes() + 1; i++) {
        //     std::cout << requisicao->get_source_offset(i) << "  ";
        // }
        //
        // std::cout << std::endl << std::endl << "quantidade de arestas: " << requisicao->get_num_edges() << std::endl << "destination indices: ";
        // for (int i = 0; i < requisicao->get_num_edges(); i++) {
        //     std::cout << requisicao->get_destination_indice(i) << "  ";
        // }
        // std::cout << std::endl << std::endl ;

        return requisicao;
    }
  }
}

#endif
