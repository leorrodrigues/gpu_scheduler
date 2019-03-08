#ifndef _ALLOCATION_MODELS_CUH
#define _ALLOCATION_MODELS_CUH

/*! \file
 *  \brief Alocaiion Models
 */

#include <vnegpu/graph.cuh> //Main data structure
#include <vnegpu/algorithm/mcl.cuh> //MCL algorithm
#include <vnegpu/algorithm/fit.cuh> //Fit algorithm
#include <vnegpu/allocation_policies.cuh> //Allocation polocies used in the fit algorithm
#include <vnegpu/util/group.cuh>
#include <vnegpu/algorithm/r_kleene.cuh>
#include <vnegpu/algorithm/k_means.cuh>

namespace vnegpu
{
  namespace allocation
  {

    enum models{
      MODEL_1,
      MODEL_2,
      MODEL_3,
      MODEL_4
    };

    template <typename T, class VariablesType, class AllocationPolice>
    vnegpu::fit_return fit_model_1(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, AllocationPolice police, int id=0){
      data_center->update_gpu(true);
      request->update_gpu();

      vnegpu::fit_return done = vnegpu::algorithm::fit(data_center, request, police);

      if(done!=vnegpu::FIT_SUCCESS){
        return done;
      }

      data_center->update_cpu();
      request->update_cpu();

      return done;
    }

    template <typename T, class VariablesType, class AllocationPolice>
    void desfit_model_1(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, AllocationPolice police){
        vnegpu::algorithm::detail::desalloc_imp(data_center, request, police);
    }

    template <typename T, class VariablesType, class AllocationPolice1, class AllocationPolice2>
    vnegpu::fit_return fit_model_2(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, AllocationPolice1 police1, AllocationPolice2 police2, bool run_group=true){
      //Update both graphs to GPU.
      data_center->update_gpu(true);
      request->update_gpu();

      if(run_group){
          vnegpu::algorithm::mcl(data_center, data_center->variables.edge_band, 2, 1.2, 0.1);
          //vnegpu::algorithm::r_kleene(data_center, data_center->variables.edge_band);
          //vnegpu::algorithm::k_means(data_center, 70, vnegpu::distance::matrix_based());
      }

      vnegpu::graph<T, VariablesType>* data_center_agruped = vnegpu::util::create_graph_from_group(data_center, true);

      vnegpu::fit_return done = vnegpu::algorithm::fit(data_center_agruped, request, police1, true);
      if(done!=vnegpu::FIT_SUCCESS){
        data_center_agruped->free_graph();
        delete data_center_agruped;
        return done;
      }


      done = vnegpu::algorithm::fit(data_center, request, police2);
      if(done!=vnegpu::FIT_SUCCESS){
        data_center_agruped->free_graph();
        delete data_center_agruped;
        return done;
      }

      //Update both graphs to CPU.
      data_center->update_cpu(true);
      request->update_cpu();
      data_center_agruped->free_graph();
      delete data_center_agruped;
      return done;
    }

    template <typename T, class VariablesType, class AllocationPolice1>
    void desfit_model_2(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, AllocationPolice1 police){
        vnegpu::algorithm::detail::desalloc_imp(data_center, request, police);
    }


    template <typename T, class VariablesType, class AllocationPolice1>
    vnegpu::fit_return fit_model_3(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, AllocationPolice1 police){
      //Update both graphs to GPU.
      data_center->update_gpu(true);
      request->update_gpu();

      vnegpu::algorithm::mcl(request, request->variables.edge_band, 2, 1.2, 0.1);
      //vnegpu::algorithm::mcl(request, request->variables.edge_band, 2, 1.3, 0.000001);


      vnegpu::graph<T, VariablesType>* request_agruped = vnegpu::util::create_graph_from_group(request);

      request->set_group_graph(request_agruped);

      vnegpu::fit_return done = vnegpu::algorithm::fit(data_center, request_agruped, police);

      if(done!=vnegpu::FIT_SUCCESS){
        return done;
      }

      vnegpu::util::map_request_gruped_allocation(request, request_agruped);

      //Update both graphs to CPU.
      data_center->update_cpu();
      request->update_cpu();
      request_agruped->update_cpu();

      return done;

    }

    template <typename T, class VariablesType, class AllocationPolice1>
    void desfit_model_3(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, AllocationPolice1 police){
        vnegpu::algorithm::detail::desalloc_imp(data_center, request->get_group_graph(), police);
    }

    template <typename T, class VariablesType, class AllocationPolice1, class AllocationPolice2>
    vnegpu::fit_return fit_model_4(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, AllocationPolice1 police1, AllocationPolice2 police2, int id, bool run_group=true){
      //Update both graphs to GPU.
      data_center->update_gpu(true);
      request->update_gpu(true);
      vnegpu::fit_return done;

      vnegpu::algorithm::mcl(request, request->variables.edge_band, 2, 1.2, 0.1);

      vnegpu::graph<T, VariablesType>* request_agruped = vnegpu::util::create_graph_from_group(request, true);

      request->set_group_graph(request_agruped);

      if(run_group){
          //vnegpu::algorithm::r_kleene(data_center, data_center->variables.edge_band);
          //vnegpu::algorithm::k_means(data_center, 70, vnegpu::distance::matrix_based());
          vnegpu::algorithm::mcl(data_center, data_center->variables.edge_band, 2, 1.2, 0.1);
      }

      vnegpu::graph<T, VariablesType>* data_center_agruped = vnegpu::util::create_graph_from_group(data_center, true);

      done = vnegpu::algorithm::fit(data_center_agruped, request_agruped, police1, true);
      if(done!=vnegpu::FIT_SUCCESS){
        request_agruped->free_graph();
        data_center_agruped->free_graph();
        delete data_center_agruped;
        delete request_agruped;
        return done;
      }

      vnegpu::util::map_request_gruped_allocation(request, request_agruped);

      done = vnegpu::algorithm::fit(data_center, request, police2);

      if(done!=vnegpu::FIT_SUCCESS){
        request_agruped->free_graph();
        data_center_agruped->free_graph();
        delete data_center_agruped;
        delete request_agruped;
        return done;
        //return false;
      }

      //Update both graphs to CPU.
      data_center->update_cpu(true);
      request->update_cpu(true);
      request_agruped->free_graph();
      data_center_agruped->free_graph();
      delete data_center_agruped;
      delete request_agruped;

      return done;
    }

    template <typename T, class VariablesType, class AllocationPolice1>
    void desfit_model_4(graph<T,VariablesType> *data_center, graph<T,VariablesType> *request, AllocationPolice1 police){
        vnegpu::algorithm::detail::desalloc_imp(data_center, request, police);
    }

  }
}

#endif
