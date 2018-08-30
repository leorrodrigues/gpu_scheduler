#ifndef _POLITICA_ALOCACAO_GRUPOS_CUH
#define _POLITICA_ALOCACAO_GRUPOS_CUH

/*! \file
 *  \brief Alocaiion Functors
 */

#include <iostream>
#include <cmath>
#include <vnegpu/graph.cuh>
#include <vnegpu/metrics.cuh>
#include <vnegpu/algorithm/generic_rank.cuh>
#include <vnegpu/algorithm/local_metric.cuh>
// #include <vnegpu/util/funcao_beta.cuh>

#define IND_VAR_PROBABILIDADE_FALHA 4
#define IND_VAR_CUSTO_VM 5
#define IND_VAR_GRUPO_CUSTO 6

namespace vnegpu
{
  namespace allocation
  {

     struct worst_fit_agrupado
     {
       int total_grupos_p = 0;
       int total_grupos_c = 0;
       int grupos_usados_p = 0;
       int grupos_usados_c = 0;
       int ind_menor_custo_alocacao;
       bool tipo_certo = false;
       bool capacidade_suficiente = false;
       bool grupo_certo_p;
       bool grupo_certo_c;

       worst_fit_agrupado(int total_grupos_probabilidade = 0, int grupos_probabilidade = 0, int total_grupos_custo = 0, int grupos_custo = 0){
           total_grupos_p = total_grupos_probabilidade;
           grupos_usados_p = grupos_probabilidade;
           total_grupos_c = total_grupos_custo;
           grupos_usados_c = grupos_custo;
           // printf("%d\n", total_grupos_p);
           grupo_certo_p = total_grupos_p == 0;
           grupo_certo_c = total_grupos_c == 0;
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_best_node(graph<T,VariablesType> *provedores, int a, int b)
       {
         int dif_capacidade = provedores->get_variable_node(provedores->variables.node_capacity, a) - provedores->get_variable_node(provedores->variables.node_capacity, b);
         if(dif_capacidade > 0){
             return true;
         }else if(dif_capacidade == 0){
             bool menor_custo_alocacao = provedores->get_variable_node(IND_VAR_CUSTO_VM, a) <= provedores->get_variable_node(IND_VAR_CUSTO_VM, b);
             bool menor_probabilidade_falha = provedores->get_variable_node(IND_VAR_PROBABILIDADE_FALHA, a) <= provedores->get_variable_node(IND_VAR_PROBABILIDADE_FALHA, b);
             return menor_probabilidade_falha && menor_custo_alocacao;
         }
         return false;
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_node_allocable(graph<T,VariablesType> *provedores, graph<T,VariablesType> *requisicao, int host_id, int requisicao_id)
       {
         tipo_certo = provedores->get_node_type(host_id) == vnegpu::TYPE_HOST || provedores->get_node_type(host_id) == requisicao->get_node_type(requisicao_id);
         capacidade_suficiente = provedores->get_variable_node(provedores->variables.node_capacity, host_id) >= requisicao->get_variable_node(requisicao->variables.node_capacity, requisicao_id);

         for (int i = 0; i < total_grupos_p; i++) {
             int valor_elevado = pow(2, i);
             if((grupos_usados_p & valor_elevado) == valor_elevado){
                if(grupo_certo_p = (provedores->get_group_id(host_id) == i)) break;

             }
         }

         for (int i = 0; i < total_grupos_c; i++) {
             int valor_elevado = pow(2, i);
             if((grupos_usados_c & valor_elevado) == valor_elevado){
                if(grupo_certo_c = (provedores->get_variable_node(IND_VAR_GRUPO_CUSTO, host_id) == i)) break;
             }
         }
         // printf("%d\n", grupo_certo_p);

         return tipo_certo && capacidade_suficiente && grupo_certo_p && grupo_certo_c;
       }

       template <typename T, class VariablesType>
       __host__ __device__
       bool is_edge_allocable(graph<T,VariablesType> *provedores, graph<T,VariablesType> *requisicao, int host_id, int requisicao_id)
       {
         return provedores->get_variable_edge(provedores->variables.edge_capacity, host_id) >= requisicao->get_variable_edge(requisicao->variables.edge_capacity, requisicao_id);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_edge(graph<T,VariablesType> *provedores, graph<T,VariablesType> *requisicao, int host_id, int requisicao_id)
       {
         T new_value = provedores->get_variable_edge(provedores->variables.edge_capacity, host_id)-requisicao->get_variable_edge(requisicao->variables.edge_capacity, requisicao_id);
         provedores->set_variable_edge(provedores->variables.edge_capacity, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void alloc_node(graph<T,VariablesType> *provedores, graph<T,VariablesType> *requisicao, int host_id, int requisicao_id)
       {
         T new_value = provedores->get_variable_node(provedores->variables.node_capacity, host_id)-requisicao->get_variable_node(requisicao->variables.node_capacity, requisicao_id);
         provedores->set_variable_node(provedores->variables.node_capacity, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_edge(graph<T,VariablesType> *provedores, graph<T,VariablesType> *requisicao, int host_id, int requisicao_id)
       {
         T new_value = provedores->get_variable_edge(provedores->variables.edge_capacity, host_id)+requisicao->get_variable_edge(requisicao->variables.edge_capacity, requisicao_id);
         provedores->set_variable_edge(provedores->variables.edge_capacity, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       void desalloc_node(graph<T,VariablesType> *provedores, graph<T,VariablesType> *requisicao, int host_id, int requisicao_id)
       {
         T new_value = provedores->get_variable_node(provedores->variables.node_capacity, host_id)+requisicao->get_variable_node(requisicao->variables.node_capacity, requisicao_id);
         provedores->set_variable_node(provedores->variables.node_capacity, host_id, new_value);
       }

       template <typename T, class VariablesType>
       __host__ __device__
       T inline edge_distance(vnegpu::graph<T,VariablesType> *graph, int node1_id, int node2_id){
         return (T)1;
       }

       template <typename T, class VariablesType>
       __host__
       void inline node_each_iteration(vnegpu::graph<T,VariablesType> *provedores, vnegpu::graph<T,VariablesType> *requisicao){

       }

     };

  }
}

#endif
