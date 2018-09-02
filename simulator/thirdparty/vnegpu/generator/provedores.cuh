#ifndef _PROVEDORES_CUH
#define _PROVEDORES_CUH

/*! \file
 *  \brief The generator of bcubes.
 */

// #include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vnegpu/graph.cuh>
#include <vnegpu/util/util.cuh>

namespace vnegpu
{
  namespace generator
  {

    template <typename T, class VariablesType=vnegpu::graph_type::minimalist>
    vnegpu::graph<T,VariablesType>* provedores(){

        std::ifstream arquivoProvedores;
        arquivoProvedores.open("provedores.txt", std::fstream::in);

        //conta quantas linhas e consequentemente zonas há no arquivo
        int qtd_zonas = std::count(std::istreambuf_iterator<char>(arquivoProvedores), std::istreambuf_iterator<char>(), '\n');
        // std::cout << qtd_zonas << std::endl;

        //quantidade de links entre zonas, cada zona se comunica com todas as outras 224, logo 225x224 links.
        int qtdLinks = ((qtd_zonas - 1) * qtd_zonas)/2;

        //retorna o ponteiro do arquivo para o começo do mesmo
        arquivoProvedores.clear();
        arquivoProvedores.seekg(0, std::ios::beg);

        std::vector<std::string> valores_entrada;
        std::vector<std::vector<double> > custo_links(qtd_zonas, std::vector<double>(3));
        //cria o grafo
        vnegpu::graph<T, VariablesType>* provedores = new vnegpu::graph<T, VariablesType>(qtd_zonas, qtdLinks, VariablesType::total_nodes_variables, VariablesType::total_edges_variables);

        //criação das variáveis referentes aos nós e as arestas, contendo as informações necessárias
        int ind_var_provedor = provedores->add_node_variable("Provedor");
        int ind_var_regiao = provedores->add_node_variable("Região");
        int ind_var_zona = provedores->add_node_variable("Zona");
        int ind_var_probabilidade_falha = provedores->add_node_variable("Probablidade de falha");
        int ind_var_custo_vm = provedores->add_node_variable("CustoAlocacaoVM");
        int ind_var_dst_probabilidade = provedores->add_edge_variable("DistanciaProbabilidadeFalha");
        provedores->add_node_variable("MelhorNo");

        //variáveis para controlar a leitura da região e do provedor
        int ind_provedor = 0;
        int ind_regiao = 0;
        std::string provedor_anterior;
        std::string regiao_anterior;

        //percorre todas as linhas do arquivo, lendo as informações de cada zona
        for (int i = 0; i < qtd_zonas; i++) {

            //copia uma linha do arquivo para a variável "linha" e então percorre a string até achar um "/t",
            //colocando a parte lida na posição respectiva do vetor de strings
            //para ao terminar a linha
            std::string linha;
            int indice_anterior = 0;
            std::getline(arquivoProvedores, linha);
            for(std::string::size_type j = 0; j < linha.size(); ++j)
            {
                if(linha[j] == '\t' || (j == linha.size() - 1)) {
                    valores_entrada.push_back(linha.substr(indice_anterior, j - indice_anterior));
                    indice_anterior = j + 1;
                }
            }

            //se o valor referente ao provedor no vetor de strings for diferente do anterior modifica o indice de provedor
            if(valores_entrada[0] != provedor_anterior){
                     ind_provedor++;
                     ind_regiao = 0;
            }

            //aplica a mesma lógica do indice de provedor, desta vez para o indice de região
            if(valores_entrada[1] != regiao_anterior) ind_regiao++;

            //calcula a probabilidade de falha baseado no cálculo do artigo
            int qtd_falhas = std::stoi(valores_entrada[3], nullptr);
            float duracao_falhas = std::stof(valores_entrada[4], nullptr);

            float mtbf = 0;
            float probabilidade_falha = 0;
            if(qtd_falhas != 0) mtbf = (720 - duracao_falhas)/qtd_falhas;
            if(mtbf != 0) probabilidade_falha = (1/mtbf);

            //converte a string que contém o índice da zona para float
            float a = std::stof(valores_entrada[2], nullptr);


            //popula as variáveis dos nós com as informações lidas do arquivo, devidamente convertidas para float.
            provedores->set_variable_node(ind_var_provedor, i, (float)ind_provedor);
            provedores->set_variable_node(ind_var_regiao, i, (float)ind_regiao);
            provedores->set_variable_node(ind_var_zona, i, std::stof(valores_entrada[2], nullptr));
            provedores->set_variable_node(ind_var_probabilidade_falha, i, probabilidade_falha);
            provedores->set_variable_node(ind_var_custo_vm, i, std::stof(valores_entrada[5], nullptr));

            //guarda os custos de links para cada uma das zonas de modo a povoar as arestas depois
            for(int j = 0; j < 3; custo_links[i][j] = std::stod(valores_entrada[6 + j++], nullptr))

            //seta o source offset referente ao nó atual
            provedores->set_source_offsets(i, i*(qtd_zonas - 1));
            provedores->set_node_type(i, vnegpu::TYPE_HOST);

            //atualiza os últimos valores lidos para provedor e região
            provedor_anterior = valores_entrada[0];
            regiao_anterior = valores_entrada[1];

            // limpa o vetor de entrada
            valores_entrada.clear();

            // preenche os destination indices do nó em questão
            for (int j = 0; j < qtd_zonas; j++) {
                int correcao_autolink = 0;
                if(j > i) correcao_autolink = 1;
                if(j != i){
                     provedores->set_destination_indices(i*(qtd_zonas - 1) + j - correcao_autolink, j);
                }
            }
        }

        provedores->set_source_offsets(qtd_zonas, qtdLinks*2);
        provedores->check_edges_ids();
        provedores->set_hosts(qtd_zonas);
        // for (size_t i = 0; i < custo_links.size(); i++) {
        //     std::cout << custo_links[i][0] << " " << custo_links[i][1] << " " << custo_links[i][2] << std::endl;
        // }

        for (int i = 0; i < qtd_zonas; i++) {
            for (int j = 0; j < qtd_zonas; j++) {
                int correcao_autolink = 0;
                if(j > i) correcao_autolink = 1;
                //verifica se as duas zonas se encontram no mesmo provedor
                if(provedores->get_variable_node(ind_var_provedor, i) == provedores->get_variable_node(ind_var_provedor, j)){
                    //verifica se as duas zonas se encontram na mesma região
                    if(provedores->get_variable_node(ind_var_regiao, i) == provedores->get_variable_node(ind_var_regiao, j)){
                        //preço de link entre zonas na mesma região
                        provedores->set_variable_edge(provedores->variables.edge_capacity, i*(qtd_zonas - 1) + j - correcao_autolink, provedores->get_variable_edge(provedores->variables.edge_capacity, i*(qtd_zonas - 1) + j - correcao_autolink) + custo_links[i][2]);
                    }else{
                        //preço de link entre zonas no mesmo provedor
                        if((i == 41 && j == 29) || (i == 29 && j == 41)) std::cout << provedores->get_variable_edge(provedores->variables.edge_capacity, i*(qtd_zonas - 1) + j - correcao_autolink) << std::endl;
                        provedores->set_variable_edge(provedores->variables.edge_capacity, i*(qtd_zonas - 1) + j - correcao_autolink, provedores->get_variable_edge(provedores->variables.edge_capacity, i*(qtd_zonas - 1) + j - correcao_autolink) + custo_links[i][1]);
                    }
                }else{
                    //preço de link entre zonas em provedores distintos
                    provedores->set_variable_edge(provedores->variables.edge_capacity, i*(qtd_zonas - 1) + j - correcao_autolink, provedores->get_variable_edge(provedores->variables.edge_capacity, i*(qtd_zonas - 1) + j - correcao_autolink) + custo_links[i][0]);
                }
                // if(i == 1) std::cout << i << ", " << j << " p1: " << provedores->get_variable_node(ind_var_provedor, i) << " p2: " << provedores->get_variable_node(ind_var_provedor, j) << " r1: " << provedores->get_variable_node(ind_var_regiao, i) << " r2: " << provedores->get_variable_node(ind_var_regiao, j) << ": " << provedores->get_variable_edge(ind_var_custo_link, i*(qtd_zonas - 1) + j - correcao_autolink) << std::endl;
                float distancia_probabilidade_falha = abs(provedores->get_variable_node(ind_var_probabilidade_falha, i) - provedores->get_variable_node(ind_var_probabilidade_falha, j)) * 100000 + 1;
                provedores->set_variable_edge(ind_var_dst_probabilidade, i*(qtd_zonas - 1) + j - correcao_autolink, distancia_probabilidade_falha);
            }
        }
      return provedores;
    }

  }
}
#endif
