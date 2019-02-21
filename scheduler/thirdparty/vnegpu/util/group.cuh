#ifndef _GROUP_CUH
#define _GROUP_CUH

#include <vnegpu/graph.cuh>

#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/device_ptr.h>

/*! \file
 *  \brief Extra Functions for Group Pre-Processing
 */
 namespace vnegpu
 {
   namespace util
   {

     template <typename T, class VariablesType>
     __global__
     void normalize_group_ids(vnegpu::graph<T,VariablesType> original,
                              int* ref)
     {
       int id = threadIdx.x + blockIdx.x * blockDim.x;

       if(id >= original.get_num_nodes())
            return;

       int group = ref[original.get_group_id(id)];
       original.set_group_id(id, group);
     }

     __global__ void map_group_ids(int num_groups,
                              int* unique,
                              int* ref)
     {
       for(int i=0;i<num_groups;i++)
       {
          ref[unique[i]]=i;
       }
     }

     template <typename T, class VariablesType>
     __global__ void
     verify_group_edges(vnegpu::graph<T,VariablesType> original,
                        int* source_ref,
                        int number_groups,
                        vnegpu::util::matrix<int> connections)
     {
       int id = threadIdx.x + blockIdx.x * blockDim.x;


       if(id >= original.get_num_nodes())
            return;

         for(int i=original.get_source_offset(id); i<original.get_source_offset(id+1); i++)
         {
            int neighbor = original.get_destination_indice(i);
            int source_group = original.get_group_id(id);
            int neighbor_group = original.get_group_id(neighbor);
            if(source_group != neighbor_group)
            {
              if( !atomicCAS(connections.get_element_poiter(source_group, neighbor_group), 0, 1) )
              {
                atomicAdd(&source_ref[source_group], 1);
                atomicAdd(&source_ref[number_groups], 1);
              }
            }
         }
     }

     template <typename T, class VariablesType>
     __global__ void
     set_destinations(vnegpu::graph<T,VariablesType> new_graph,
                        vnegpu::util::matrix<int> connections)
     {
       int id = threadIdx.x + blockIdx.x * blockDim.x;

       if(id >= new_graph.get_num_nodes())
            return;

       int this_offset = new_graph.get_source_offset(id);

       for(int i=0; i<new_graph.get_num_nodes(); i++)
       {
          if(connections.get_element(id, i)){
            connections.set_element(id, i, this_offset);
            new_graph.set_destination_indices(this_offset++, i);
          }
       }

     }

     template <typename T, class VariablesType>
     __global__ void
     edges_combine(vnegpu::graph<T,VariablesType> original,
                 vnegpu::util::matrix<int> connections,
                 vnegpu::graph<T,VariablesType> new_graph)
     {
       int id = threadIdx.x + blockIdx.x * blockDim.x;


       if(id >= original.get_num_nodes())
            return;

         for(int i=original.get_source_offset(id); i<original.get_source_offset(id+1); i++)
         {
            int neighbor = original.get_destination_indice(i);
            int source_group = original.get_group_id(id);
            int neighbor_group = original.get_group_id(neighbor);
            if(id < neighbor && source_group != neighbor_group)
            {
              int offset = connections.get_element(source_group, neighbor_group);
              for(int y=0; y<original.get_num_var_edges(); y++){
                T v = original.get_variable_edge(y, i);
                T* p = new_graph.get_variable_edge_pointer(y, offset);
                atomicAdd(p, v);
              }
            }
         }
     }

     template <typename T, class VariablesType>
     __global__ void
     nodes_combine(vnegpu::graph<T,VariablesType> original,
                   vnegpu::graph<T,VariablesType> new_graph,
                   bool check_max)
     {
       int id = threadIdx.x + blockIdx.x * blockDim.x;

       if(id >= original.get_num_nodes())
            return;

        int source_group = original.get_group_id(id);

        for(int y=0; y<original.get_num_var_nodes(); y++){
          T v = original.get_variable_node(y, id);
          T* p = new_graph.get_variable_node_pointer(y, source_group);
          atomicAdd(p, v);
          if(original.get_node_type(id) == vnegpu::TYPE_HOST){
            new_graph.set_node_type(source_group, vnegpu::TYPE_HOST);
          }

        }
        if(check_max){
          //TODO
          //T v = original.get_variable_node(original.variables.node_cpu, id);
          //T* p = new_graph.get_variable_node_pointer(new_graph.variables.node_rank, source_group);
          //atomicMax(p, v);
        }

     }

     template <typename T, class VariablesType>
     __global__ void
     map_request_gruped_allocation_kernel(vnegpu::graph<T,VariablesType> request,
                                          vnegpu::graph<T,VariablesType> request_gruped)
     {
       int id = threadIdx.x + blockIdx.x * blockDim.x;

       if(id >= request.get_num_nodes())
            return;

        int group = request.get_group_id(id);
        int alloc_to = request_gruped.get_allocation_to_nodes_ids(group);
        request.set_allocation_to_nodes_ids(id, alloc_to);

     }

     template <typename T, class VariablesType>
     __host__ void
     map_request_gruped_allocation(vnegpu::graph<T,VariablesType>* request, vnegpu::graph<T,VariablesType>* request_gruped)
     {
       request->initialize_allocation();

       int num = request->get_num_nodes()/CUDA_BLOCK_SIZE + 1;

       dim3 Block(CUDA_BLOCK_SIZE);
       dim3 Grid(num);

       map_request_gruped_allocation_kernel<<<Grid, Block>>>(*request, *request_gruped);
     }

     template <typename T, class VariablesType>
     __host__ vnegpu::graph<T,VariablesType>*
     create_graph_from_group(vnegpu::graph<T,VariablesType>* original, bool check_max=false)
     {
       int* sort;
       int* unique;
       original->update_cpu();
       int num_nodes = original->get_num_nodes();

       cudaMalloc(&sort, sizeof(int)*num_nodes);
       cudaMalloc(&unique, sizeof(int)*num_nodes);

       int* grupos = (int*)malloc(sizeof(int)*num_nodes);
       for(int i=0;i<num_nodes;i++){
         grupos[i]=-1;
       }

       cudaMemcpy(sort, original->get_group_d_ptr(), sizeof(int)*num_nodes, cudaMemcpyDeviceToDevice);

       thrust::device_ptr<int> dev_sort = thrust::device_pointer_cast(sort);
       thrust::device_ptr<int> dev_unique = thrust::device_pointer_cast(unique);

       thrust::sort(dev_sort, dev_sort + num_nodes);

       thrust::device_ptr<int> result_end = thrust::unique_copy(dev_sort, dev_sort + num_nodes, dev_unique);

       int number_groups = result_end-dev_unique;
       //printf("F:%d\n",number_groups);

    /*   int num_groups_2 = 0;
       for(int i=0;i<num_nodes;i++){
         int x = original->get_group_id(i);
         if(x>=num_nodes || x<0){
           printf("ERRO:%d\n",x);
           continue;
         }
         if(grupos[x]==-1){
           grupos[x]=num_groups_2;
           num_groups_2++;
         }
       }
       printf(">%d\n",num_groups_2);
       if(number_groups==6){
         exit(0);
       }*/


       map_group_ids<<<1,1>>>(number_groups, unique, sort);

       int num = original->get_num_nodes()/CUDA_BLOCK_SIZE + 1;

       dim3 Block(CUDA_BLOCK_SIZE);
       dim3 Grid(num);

       normalize_group_ids<<<Grid, Block>>>(*original, sort);

       int* source_ref;
       cudaMalloc(&source_ref, sizeof(int)*(number_groups+1) );
       cudaMemset(source_ref, 0, sizeof(int)*(number_groups+1) );

       vnegpu::util::matrix<int>* connections = new vnegpu::util::matrix<int>(number_groups, number_groups);

       cudaMemset(connections->data, 0, number_groups*connections->pitch );

       verify_group_edges<<<Grid, Block>>>(*original, source_ref, number_groups, *connections);

       int* new_source = (int*)malloc(sizeof(int)*(number_groups+1));

       cudaMemcpy(new_source, source_ref, sizeof(int)*(number_groups+1), cudaMemcpyDeviceToHost);

       //for(int i=0; i<=number_groups; i++){
      //   printf("Source[%d]=%d\n", i, new_source[i]);
       //}
       vnegpu::graph<T,VariablesType>* new_graph = new vnegpu::graph<T,VariablesType>(
                                                        number_groups,
                                                        new_source[number_groups],
                                                        original->get_num_var_nodes(),
                                                        original->get_num_var_edges());
       int offset = 0;

       for(int id=0; id<number_groups;id++)
       {
         new_graph->set_source_offsets(id, offset);
         new_graph->set_node_type(id, vnegpu::TYPE_SWITH);//neutral element
         for(int variable=0; variable<new_graph->get_num_var_nodes(); variable++){
           new_graph->set_variable_node(variable, id, 0.0f);
         }
         offset+=new_source[id];
       }

       for(int id=0; id<new_source[number_groups]; id++){
         for(int variable=0; variable<new_graph->get_num_var_edges(); variable++){
           new_graph->set_variable_edge_undirected(variable, id, 0.0f);
         }
       }
       new_graph->set_source_offsets(number_groups, new_source[number_groups]);

       new_graph->update_gpu();

       int num2 = new_graph->get_num_nodes()/CUDA_BLOCK_SIZE + 1;

       dim3 Block2(CUDA_BLOCK_SIZE);
       dim3 Grid2(num);

       set_destinations<<<Grid2, Block2>>>(*new_graph, *connections);

       new_graph->update_cpu();

       new_graph->check_edges_ids();

       new_graph->update_gpu();

       edges_combine<<<Grid, Block>>>(*original, *connections, *new_graph);

       nodes_combine<<<Grid, Block>>>(*original, *new_graph, check_max);

       connections->free();
       delete connections;
       cudaFree(sort);
       cudaFree(unique);
       cudaFree(source_ref);
       free(grupos);

       return new_graph;

     }
   }
}
#endif
