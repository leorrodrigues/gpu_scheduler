#ifndef _FIT_IMP_CUH
#define _FIT_IMP_CUH

/*! \file
 *  \brief Fit Core/Implementation functions
 */

#include <limits>

#include <vnegpu/graph.cuh>
#include <vnegpu/util/util.cuh>



namespace vnegpu
{
  namespace algorithm
  {
    namespace detail
    {

      template <typename T, class VariablesType, class AlocationFunctor>
      __global__
      void initialize_dijkstra_with_check(vnegpu::graph<T,VariablesType> graph_distance,
                                          graph<T,VariablesType> request,
                                          int edge_request,
                                          T* distance_index,
                                          T* d_distance_temp,
                                          bool* d_frontier,
                                          T infinite,
                                          int initial_node,
                                          bool* valid,
                                          int* path,
                                          AlocationFunctor func)
      {
        int id = threadIdx.x + blockIdx.x * blockDim.x;

        if(id >= graph_distance.get_num_nodes())
             return;

        if(request.get_allocation_to_nodes_ids(initial_node)==id)
        {
          distance_index[id]=0;
          d_distance_temp[id]=0;
          d_frontier[id]=1;
          path[id]=-1;
        }else{
          distance_index[id]=infinite;
          d_distance_temp[id]=infinite;
          d_frontier[id]=0;
          path[id]=-1;
        }

        for(int i=graph_distance.get_source_offset(id); i<graph_distance.get_source_offset(id+1); i++)
        {
            valid[i]=func.is_edge_allocable(&graph_distance, &request, i, edge_request);
        }

      }

      template <typename T, class VariablesType, class AlocationFunctor>
      __global__
      void dijkstra_iteration_with_check(vnegpu::graph<T,VariablesType> graph_distance,
                                         T* d_distance_temp,
                                         bool* d_frontier,
                                         bool* d_done,
                                         AlocationFunctor alloc,
                                         T* distance_index,
                                         bool* valid)
      {
        int id = threadIdx.x + blockIdx.x * blockDim.x;

        if(id==0){
          *d_done = true;
        }

        if(id >= graph_distance.get_num_nodes())
             return;

        if(d_frontier[id])
        {
          d_frontier[id]=false;
          T self_distance = distance_index[id];

          for(int i=graph_distance.get_source_offset(id); i<graph_distance.get_source_offset(id+1); i++)
          {
              if(valid[i])
              {
                T distance = alloc.edge_distance(&graph_distance,i,id);
                int neighbor = graph_distance.get_destination_indice(i);
                atomicMin(&d_distance_temp[neighbor], self_distance+distance);
              }
          }
        }

      }

      template <typename T, class VariablesType, class AlocationFunctor>
      __global__ void dijkstra_save_with_path(vnegpu::graph<T,VariablesType> graph_distance, T* d_distance_temp, bool* d_frontier, bool* d_done, T* distance_index, int* path, AlocationFunctor alloc, bool* valid)
      {
        int id = threadIdx.x + blockIdx.x * blockDim.x;

        if(id >= graph_distance.get_num_nodes())
             return;

        T self_distance = distance_index[id];

        if(self_distance > d_distance_temp[id]){


          d_frontier[id]=true;
          *d_done = false;
          for(int i=graph_distance.get_source_offset(id); i<graph_distance.get_source_offset(id+1); i++)
          {
            if(valid[i]){
              T distance = alloc.edge_distance(&graph_distance,i,id);
              int neighbor = graph_distance.get_destination_indice(i);
              if((distance+d_distance_temp[neighbor])==d_distance_temp[id])
              {
                //printf("saving[%d]=%d\n",id,i);
                path[id]=i;
                break;
              }
            }
          }
          distance_index[id]=d_distance_temp[id];
        }
        d_distance_temp[id] = distance_index[id];

      }


      template <typename T, class VariablesType, class AlocationFunctor>
      void dijkstra_imp_with_allocation_check(graph<T,VariablesType> *graph_distance,
                                              int initial_node,
                                              graph<T,VariablesType> *request,
                                              int edge_request,
                                              T* d_temp_distance,
                                              int* d_path,
                                              AlocationFunctor alloc,
                                              T* d_distance_temp,
                                              bool* d_frontier,
                                              bool* d_valid,
                                              bool* d_done)
      {
        int num = graph_distance->get_num_nodes()/CUDA_BLOCK_SIZE + 1;

        dim3 Block(CUDA_BLOCK_SIZE);
        dim3 Grid(num);

        T infinite = std::numeric_limits<T>::max();



        initialize_dijkstra_with_check<<<Grid, Block>>>(*graph_distance, *request, edge_request, d_temp_distance, d_distance_temp, d_frontier, infinite, initial_node, d_valid, d_path, alloc);
        CUDA_CHECK();
        bool done = false;

        int i=100;
        while(!done && i--){
          dijkstra_iteration_with_check<<<Grid, Block>>>(*graph_distance, d_distance_temp, d_frontier, d_done, alloc, d_temp_distance, d_valid);
          CUDA_CHECK();
          dijkstra_save_with_path<<<Grid, Block>>>(*graph_distance, d_distance_temp, d_frontier, d_done, d_temp_distance, d_path, alloc, d_valid);
          CUDA_CHECK();
          cudaMemcpy(&done,d_done,sizeof(bool),cudaMemcpyDeviceToHost);
        }



      }

      template <typename T, class VariablesType, class AlocationFunctor>
      __global__ void alloc_node(graph<T,VariablesType> data_center,
                                graph<T,VariablesType> request,
                                int request_id,
                                AlocationFunctor func,
                                int num,
                                bool* error)
      {
        int tx = threadIdx.x;
        int id = tx*num;
        __shared__ int best_per_thread[512];

        int my_best=-1;
        for(int i=0;i<num && (id+i)<data_center.get_num_nodes();i++){
          if(func.is_node_allocable(&data_center, &request, id+i, request_id))
          {
            if( my_best==-1 || func.is_best_node(&data_center, id+i,my_best) ){
              my_best=id+i;
            }
          }
        }
        best_per_thread[tx]=my_best;
        __syncthreads();
        for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
          if (tx < s) {
            if(best_per_thread[tx]==-1 || best_per_thread[tx+s]!=-1 && func.is_best_node(&data_center, best_per_thread[tx+s],best_per_thread[tx])){
              best_per_thread[tx]=best_per_thread[tx+s];
            }
          }
        __syncthreads();
        }

        if(tx==0){
          if(best_per_thread[0]!=-1){
            func.alloc_node(&data_center, &request, best_per_thread[0], request_id);
            request.set_allocation_to_nodes_ids(request_id, best_per_thread[0]);
            //printf("Node Request: %d allocated on Data Center Node: %d\n",request_id,best_per_thread[0]);
          }else{
            *error = true;
            //printf("\t\tNo NODE V:%f\n", request.get_variable_node(0, request_id));
          }

        }

      }

      template <typename T, class VariablesType, class AlocationFunctor>
      __global__
      void alloc_path(graph<T,VariablesType> data_center,
                     graph<T,VariablesType> request,
                     int edge_request,
                     int* path,
                     int origim,
                     int destiny,
                     AlocationFunctor func,
                     bool* error,
                     int* d_edges_allocation)
      {

        int node = request.get_allocation_to_nodes_ids(destiny);
        int real_origin = request.get_allocation_to_nodes_ids(origim);
        //verify if INF
        int edge = path[node];
        int pos_edge=0;
        d_edges_allocation[0]=-1;

        if(node==real_origin){
          //printf("SAME NODE CONDITION\n");
          return;
        }
        if(edge==-1){
          //printf("\t\tERROR ALOC NO PATH %d %d->%d\n",node,origim,destiny);
          *error=true;
          return;
        }

        int next_node = data_center.get_destination_indice(edge);
        int it=100;
        while(next_node != real_origin && it--){
          func.alloc_edge(&data_center, &request, edge, edge_request);
          d_edges_allocation[pos_edge++]=edge;
          node=next_node;
          edge = path[node];
          if(edge==-1){
            //printf("\t\tERROR ALOC NO PATH2\n");
            *error=true;
            return;
          }
          //printf("Next:%d-%d\n",next_node, edge);
          next_node = data_center.get_destination_indice(edge);
        }
        if(it<1){
          //printf("ERRAO\n");
          *error=true;
          return;
        }
        func.alloc_edge(&data_center, &request, edge, edge_request);
        d_edges_allocation[pos_edge++]=edge;
        d_edges_allocation[pos_edge++]=-1;

      }

      /**
       * \brief Classify the nodes of the graph in groups based on a distance.
       * \param graph_distance The graph to be used.
       * \param group_index Th.
       */
       template <typename T, class VariablesType, class AlocationFunctor>
       vnegpu::fit_return fit_imp(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, AlocationFunctor func, bool only_nodes)
       {

         request->initialize_allocation();

         vnegpu::fit_return ret = FIT_SUCCESS;

         int* d_path;
         cudaMalloc(&d_path, sizeof(int) * data_center->get_num_nodes() );

         int* d_edges_allocation;
         cudaMalloc(&d_edges_allocation, sizeof(int) * data_center->get_num_nodes() );
         cudaMemset(d_edges_allocation, 0, sizeof(int) * data_center->get_num_nodes());

         int* edge_allocation = (int*)malloc(sizeof(int) * data_center->get_num_nodes());

         bool error=false;

         bool* d_error;
         cudaMalloc(&d_error, sizeof(bool));

         cudaMemcpy(d_error, &error, sizeof(bool), cudaMemcpyHostToDevice);

         T* d_temp_distance;
         cudaMalloc(&d_temp_distance, sizeof(T) * data_center->get_num_edges() );

         T* d_distance_temp;
         cudaMalloc(&d_distance_temp, sizeof(T) * data_center->get_num_nodes() );

         bool* d_frontier;
         cudaMalloc(&d_frontier, sizeof(bool) * data_center->get_num_nodes() );

         bool* d_valid;
         cudaMalloc(&d_valid, sizeof(bool) * data_center->get_num_edges()*2 );
         bool* d_done;

         cudaMalloc(&d_done, sizeof(bool));

         int num = data_center->get_num_nodes()/CUDA_BLOCK_SIZE + 1;

         dim3 Block(CUDA_BLOCK_SIZE);
         dim3 Grid(1);

          for(int i=0;i<request->get_num_nodes() && !error;i++)
          {
            func.node_each_iteration(data_center, request);
            alloc_node<<<Grid, Block>>>(*data_center, *request, i, func, num, d_error);
            cudaMemcpy(&error, d_error, sizeof(bool), cudaMemcpyDeviceToHost);
          }

          if(error){
            ret = FIT_NODE_ERROR;
          }

          if(!only_nodes){

            bool* concluded_edges = (bool*)malloc(request->get_num_edges()*sizeof(bool));
            memset(concluded_edges,0,request->get_num_edges()*sizeof(bool));

            for(int id=0;id<request->get_num_nodes() && !error;id++)
            {
              for(int i=request->get_source_offset(id); i<request->get_source_offset(id+1) && !error; i++)
              {
                if(!concluded_edges[request->get_egdes_ids(i)])
                {
                  concluded_edges[request->get_egdes_ids(i)]=true;
                  int neighbor = request->get_destination_indice(i);
                  //printf("Processing Request edge %d->%d\n",id,neighbor);
                  vnegpu::algorithm::detail::dijkstra_imp_with_allocation_check(data_center, id,
                     request, i, d_temp_distance, d_path, func, d_distance_temp, d_frontier, d_valid, d_done);

                  cudaDeviceSynchronize();
                  alloc_path<<<1,1>>>(*data_center, *request, i, d_path, id, neighbor, func, d_error, d_edges_allocation);

                  cudaMemcpy(&error, d_error, sizeof(bool), cudaMemcpyDeviceToHost);
                  cudaMemcpy(edge_allocation, d_edges_allocation, sizeof(int) * data_center->get_num_nodes(), cudaMemcpyDeviceToHost);
                  int edge_pos = 0;
                  while(edge_allocation[edge_pos]!=-1){

                      request->add_allocation_to_edges_ids(i, edge_allocation[edge_pos]);

                      edge_pos++;
                  }
                  CUDA_CHECK();

                  cudaDeviceSynchronize();

                }
              }
            }

            free(concluded_edges);
          }

          if(error){
            data_center->update_gpu(true);
            if(ret==FIT_SUCCESS){
              ret=FIT_EDGE_ERROR;
            }
          }

          cudaFree(d_path);
          cudaFree(d_edges_allocation);
          cudaFree(d_distance_temp);
          cudaFree(d_frontier);
          cudaFree(d_valid);
          cudaFree(d_done);
          cudaFree(d_error);
          cudaFree(d_temp_distance);
          free(edge_allocation);

          return ret;

       }


       template <typename T, class VariablesType, class AlocationFunctor>
       void desalloc_imp(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, AlocationFunctor func)
       {

         for(int i=0;i<request->get_num_nodes();i++)
         {
           if(request->get_allocation_to_nodes_ids(i)>=data_center->get_num_nodes()){
             printf("->%d\n", request->get_allocation_to_nodes_ids(i));
             continue;
           }
           func.desalloc_node(data_center, request, request->get_allocation_to_nodes_ids(i), i);
         }

         bool* concluded_edges = (bool*)malloc(request->get_num_edges()*sizeof(bool));
         memset(concluded_edges,0,request->get_num_edges()*sizeof(bool));

         for(int id=0;id<request->get_num_nodes();id++)
         {
           for(int i=request->get_source_offset(id); i<request->get_source_offset(id+1); i++)
           {
             if(!concluded_edges[request->get_egdes_ids(i)])
             {
               concluded_edges[request->get_egdes_ids(i)]=true;

               std::vector<int>* vec_edges = request->get_allocation_to_edges_ids(i);
               for (std::vector<int>::iterator it = vec_edges->begin(); it != vec_edges->end(); it++){
                  func.desalloc_edge(data_center, request, *it, i);
               }
             }
           }
         }
         data_center->cpu_modified();
         free(concluded_edges);
       }


    }//end detail
  }//end algorithm
}//end vnegpu



#endif
