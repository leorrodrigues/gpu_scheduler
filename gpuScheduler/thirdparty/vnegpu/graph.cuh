#ifndef _GRAPH_CUH
#define _GRAPH_CUH

/*! \file
 *  \brief The main graph data structure.
 */

#include <string>
#include <vector>

#include <vnegpu/config.cuh>

#include <vnegpu/util/matrix.cuh>

#ifndef NO_XML_LIB
#include XML_LIB_INC
#endif

namespace vnegpu
{

template <typename T, class VariablesType>
class graph;

/**
 * Graph variables type stucts.
 *
 * The structs here are used to identify the id of the variables
 * on the variables arrays.
 */
namespace graph_type
{
/**
 * The minimalist graph variable type.
 *
 * Have a node cpu and an edge banding variables.
 */
struct minimalist
{
	static const unsigned int total_nodes_variables = 1;
	static const unsigned int total_edges_variables = 1;
	const unsigned int node_capacity=0; ///< Stores the id of node capacity variable.
	const unsigned int edge_capacity=0; ///< Stores the id of edge capacity variable.

	template <typename T, class VariablesType>
	__host__
	void add_variables_names(graph<T, VariablesType>* new_graph)
	{
		new_graph->add_node_variable_name("Node Capacity");
		new_graph->add_edge_variable_name("Edge Capacity");
	}
};

struct minimalist_rank
{
	static const unsigned int total_nodes_variables = 2;
	static const unsigned int total_edges_variables = 1;
	const unsigned int node_capacity=0; ///< Stores the id of node capacity variable.
	const unsigned int node_rank=1; ///< Stores the id of node capacity variable.
	const unsigned int edge_capacity=0; ///< Stores the id of edge capacity variable.

	template <typename T, class VariablesType>
	__host__
	void add_variables_names(graph<T, VariablesType>* new_graph)
	{
		new_graph->add_node_variable_name("Node Capacity");
		new_graph->add_node_variable_name("Node Rank");
		new_graph->add_edge_variable_name("Edge Capacity");
	}
};

/**
 * The normal graph variable type.
 *
 * Have a node cpu, memory, disk and an edge banding, latency variables.
 */
struct normal
{
	static const unsigned int total_nodes_variables = 3;
	static const unsigned int total_edges_variables = 2;
	const unsigned int node_cpu=0; ///< Stores the id of node cpu variable.
	const unsigned int node_memory=1; ///< Stores the id of node memory variable.
	const unsigned int node_disk=2; ///< Stores the id of node disk variable.
	const unsigned int edge_band=0; ///< Stores the id of node band variable.
	const unsigned int edge_latency=1; ///< Stores the id of node latency variable.

	template <typename T, class VariablesType>
	__host__
	void add_variables_names(graph<T, VariablesType>* new_graph)
	{
		new_graph->add_node_variable_name("Node CPU");
		new_graph->add_node_variable_name("Node Memory");
		new_graph->add_node_variable_name("Node Disk");
		new_graph->add_edge_variable_name("Edge Band");
		new_graph->add_edge_variable_name("Edge Latency");
	}
};

struct real_tests
{
	static const unsigned int total_nodes_variables = 3;
	static const unsigned int total_edges_variables = 1;
	const unsigned int node_cpu=0; ///< Stores the id of node cpu variable.
	const unsigned int node_memory=1; ///< Stores the id of node memory variable.
	const unsigned int node_rank=2; ///< Stores the id of node capacity variable.
	const unsigned int edge_band=0; ///< Stores the id of node band variable.

	template <typename T, class VariablesType>
	__host__
	void add_variables_names(graph<T, VariablesType>* new_graph)
	{
		new_graph->add_node_variable_name("Node CPU");
		new_graph->add_node_variable_name("Node Memory");
		new_graph->add_node_variable_name("Node Rank");
		new_graph->add_edge_variable_name("Edge Band");
	}
};
}  //end graph_variables_types

/**
 * Graph State on Host/Device Memory.
 * Used as a safeguard to functions restricted to GPU/CPU.
 */
enum graph_state
{
	GRAPH_ALLOCATED, ///< The Graph was Allocated on both memories.
	GRAPH_ON_GPU, ///< The Graph was last updated on/transfered to GPU, the use in CPU should be after graph#update_cpu().
	GRAPH_ON_CPU ///< The Graph was last updated on/transfered to CPU, the use in GPU should be after graph#update_gpu().
};

enum node_types
{
	TYPE_HOST,
	TYPE_SWITH,
	TYPE_SWITH_CORE
};

enum fit_return
{
	FIT_NODE_ERROR,
	FIT_EDGE_ERROR,
	FIT_SUCCESS
};

/**
 * \brief Graph Class, store graph structures and
 *  can handle memory transfers between CPU/GPU.
 *
 * The Graph is directed and stored in CSR,
 * \tparam T type for graph nodes/edges variables.
 * \tparam VariablesTypes struct that store id of variables.
 */
template <typename T, class VariablesType=vnegpu::graph_type::minimalist>
class graph
{
public:

VariablesType variables;     ///< Shotcut for variables ids.

/**
 * \brief The Graph constructor based on pre determinated size.
 *
 * Initial alocation is done on host and device.
 * @param _num_nodes number of nodes.
 * @param _num_edges number of edges.
 * @param _num_var_nodes number of nodes variables.
 * @param _num_var_edges number of edges variables.
 */
graph(unsigned int _num_nodes, unsigned int _num_edges, unsigned int _num_var_nodes, unsigned int _num_var_edges)
{
	//IF USE_NVTX is Enable this will debug on Nvidia Visual Profiller
	DEBUG_PUSH_RANGE("Graph Constructor",0);

	//Temporary handle to cancel constructor in case of max variables limit reached
	if(_num_var_nodes > GRAPH_MAX_VARIABLES || _num_var_edges > GRAPH_MAX_VARIABLES)
	{
		throw;
	}

	//Seting the new variables
	this->num_nodes=_num_nodes;
	this->num_edges=_num_edges;
	this->num_var_nodes=_num_var_nodes;
	this->num_var_edges=_num_var_edges;
	this->hosts=0;

	//Allocating the topology info in CSR.
	//this->source_offsets = (int*)malloc(sizeof(int)*(this->num_nodes+1));
	cudaMallocHost((void**)&this->source_offsets,sizeof(int)*(this->num_nodes+1));
	cudaMalloc(&this->d_source_offsets, sizeof(int)*(this->num_nodes+1));

	//this->destination_indices = (int*)malloc(sizeof(int)*this->num_edges);
	cudaMallocHost((void**)&this->destination_indices,sizeof(int)*this->num_edges*2);
	cudaMalloc(&this->d_destination_indices, sizeof(int)*this->num_edges*2);

	cudaMallocHost((void**)&this->egdes_ids,sizeof(int)*this->num_edges*2);
	cudaMalloc(&this->d_egdes_ids, sizeof(int)*this->num_edges*2);

	cudaMallocHost((void**)&this->node_type,sizeof(int)*this->num_nodes);
	cudaMalloc(&this->d_node_type, sizeof(int)*this->num_nodes);

	//Allocating info for the variables
	//this->var_nodes = (float**)malloc(sizeof(float*)*this->num_var_nodes);
	cudaMallocHost((void**)&this->var_nodes,sizeof(T*)*GRAPH_MAX_VARIABLES);
	cudaMalloc(&this->d_var_nodes, sizeof(T*)*GRAPH_MAX_VARIABLES);
	cudaMallocHost((void**)&var_nodes_d_pointes,sizeof(T*)*GRAPH_MAX_VARIABLES);

	//this->var_edges = (float**)malloc(sizeof(float*)*this->num_var_edges);
	cudaMallocHost((void**)&this->var_edges,sizeof(T*)*GRAPH_MAX_VARIABLES);
	cudaMalloc(&this->d_var_edges, sizeof(T*)*GRAPH_MAX_VARIABLES);
	cudaMallocHost((void**)&var_edges_d_pointes,sizeof(T*)*GRAPH_MAX_VARIABLES);

	//Allocating each node variable array
	for(int i=0; i<this->num_var_nodes; i++)
	{
		//this->var_nodes[i] = (float*)malloc(sizeof(float)*this->num_nodes);
		cudaMallocHost((void**)&this->var_nodes[i],sizeof(T)*this->num_nodes);
		cudaMalloc(&this->var_nodes_d_pointes[i], sizeof(T)*this->num_nodes);
	}

	//Allocating each edge variable array
	for(int i=0; i<this->num_var_edges; i++)
	{
		//this->var_edges[i] = (float*)malloc(sizeof(float)*this->num_edges);
		cudaMallocHost((void**)&this->var_edges[i],sizeof(T)*this->num_edges);
		cudaMalloc(&this->var_edges_d_pointes[i], sizeof(T)*this->num_edges);
	}

	if(this->num_var_nodes>0)
		cudaMemcpy(this->d_var_nodes, this->var_nodes_d_pointes, sizeof(T*)*this->num_var_nodes, cudaMemcpyHostToDevice);

	if(this->num_var_edges>0)
		cudaMemcpy(this->d_var_edges, this->var_edges_d_pointes, sizeof(T*)*this->num_var_edges, cudaMemcpyHostToDevice);

	state=GRAPH_ALLOCATED;

	nodes_variables_names = new std::vector<std::string>();
	edges_variables_names = new std::vector<std::string>();

	distance_matrix = NULL;
	is_grouped = false;
	is_allocated=false;

	this->variables.add_variables_names(this);

	//IF USE_NVTX is Enable this will debug on Nvidia Visual Profiller
	DEBUG_POP_RANGE();
}

/**
 * \brief Set the number of nodes.
 * \param value the new number of nodes.
 */
__host__ __device__
inline void set_num_nodes(int value)
{
	this->num_nodes=value;
}

/**
 * \brief Set the number of edges.
 * \param value the new number of edges.
 */
__host__ __device__
inline void set_num_edges(int value)
{
	this->num_edges=value;
}

/**
 * \brief Set the number of nodes.
 * \param value the new number of nodes.
 */
__host__ __device__
inline void set_hosts(int value)
{
	this->hosts=value;
}

/**
 * \brief Set the number of edges.
 * \param value the new number of edges.
 */
__host__ __device__
inline unsigned int get_hosts()
{
	return this->hosts;
}

/**
 * \brief Set the number of nodes variables.
 * \param value the new number of nodes variables.
 */
__host__ __device__
inline void set_num_var_nodes(int value)
{
	this->num_var_nodes=value;
}

/**
 * \brief Set the number of edges variables.
 * \param value the new number of edges variables.
 */
__host__ __device__
inline void set_num_var_edges(int value)
{
	this->num_var_edges=value;
}

/**
 * \brief Get the number of nodes.
 * \return number of nodes.
 */
__host__ __device__
inline unsigned int get_num_nodes()
{
	return this->num_nodes;
}

/**
 * \brief Get the number of edges.
 * \return number of edges.
 */
__host__ __device__
inline unsigned int get_num_edges()
{
	return this->num_edges;
}

/**
 * \brief Get the number of nodes variables.
 * \return number of nodes variables.
 */
__host__ __device__
inline unsigned int get_num_var_nodes()
{
	return this->num_var_nodes;
}

/**
 * \brief Get the number of edges variables.
 * \return number of edges variables.
 */
__host__ __device__
inline unsigned int get_num_var_edges()
{
	return this->num_var_edges;
}

/**
 * \brief Get the distance matrix.
 * \return matrix.
 */
__host__ __device__
inline vnegpu::util::matrix<T>* get_distance_matrix()
{
	return this->distance_matrix;
}

/**
 * \brief Set the distance matrix.
 * \param matrix matrix.
 */
__host__ __device__
inline void set_distance_matrix(vnegpu::util::matrix<T>* matrix)
{
	this->distance_matrix = matrix;
}

/**
 * \brief Set the value of a node variable.
 * \param var the id of variable array.
 * \param index the index of the node.
 * \param value the new value.
 */
__host__ __device__
inline void set_variable_node(int var, int index, T value)
{
      #ifdef __CUDA_ARCH__
	this->d_var_nodes[var][index]=value;
      #else
	this->var_nodes[var][index]=value;
      #endif
}

/**
 * \brief Get the value of a node variable.
 * \param var the id of variable array.
 * \param index the index of the node.
 * \return the variable.
 */
__host__ __device__
inline T get_variable_node(int var, int index)
{
      #ifdef __CUDA_ARCH__
	return this->d_var_nodes[var][index];
      #else
	return this->var_nodes[var][index];
      #endif
}

/**
 * \brief Set the value of a edge variable.
 * \param var the id of variable array.
 * \param index the index of the edge.
 * \param value the new value.
 */
__host__ __device__
inline void set_variable_edge(int var, int index, T value)
{

      #ifdef __CUDA_ARCH__
	this->d_var_edges[var][this->d_egdes_ids[index]]=value;
      #else
	this->var_edges[var][this->egdes_ids[index]]=value;
      #endif
}

__host__ __device__
inline T* get_variable_edge_pointer(int var, int index)
{

      #ifdef __CUDA_ARCH__
	return &this->d_var_edges[var][this->d_egdes_ids[index]];
      #else
	return &this->var_edges[var][this->egdes_ids[index]];
      #endif
}

__host__ __device__
inline T* get_variable_node_pointer(int var, int index)
{

      #ifdef __CUDA_ARCH__
	return &this->d_var_nodes[var][index];
      #else
	return &this->var_nodes[var][index];
      #endif
}

/**
 * \brief Get the value of a edge variable.
 * \param var the id of variable array.
 * \param index the index of the edge.
 * \return the variable.
 */
__host__ __device__
inline T get_variable_edge(int var, int index)
{
      #ifdef __CUDA_ARCH__
	return this->d_var_edges[var][this->d_egdes_ids[index]];
      #else
	return this->var_edges[var][this->egdes_ids[index]];
      #endif
}

/**
 * \brief Set the value of a edge variable.
 * \param var the id of variable array.
 * \param index the index of the edge.
 * \param value the new value.
 */
__host__ __device__
inline void set_variable_edge_undirected(int var, int index, T value)
{

      #ifdef __CUDA_ARCH__
	this->d_var_edges[var][index]=value;
      #else
	this->var_edges[var][index]=value;
      #endif
}

/**
 * \brief Get the value of a edge variable.
 * \param var the id of variable array.
 * \param index the index of the edge.
 * \return the variable.
 */
__host__ __device__
inline T get_variable_edge_undirected(int var, int index)
{
      #ifdef __CUDA_ARCH__
	return this->d_var_edges[var][index];
      #else
	return this->var_edges[var][index];
      #endif
}

/**
 * \brief Set the value of a edge variable.
 * \param var the id of variable array.
 * \param index the index of the edge.
 * \param value the new value.
 */
__host__ __device__
inline void set_egdes_ids(int index, T value)
{

      #ifdef __CUDA_ARCH__
	this->d_egdes_ids[index]=value;
      #else
	this->egdes_ids[index]=value;
      #endif
}

/**
 * \brief Get the value of a edge variable.
 * \param var the id of variable array.
 * \param index the index of the edge.
 * \return the variable.
 */
__host__ __device__
inline int get_egdes_ids(int index)
{
      #ifdef __CUDA_ARCH__
	return this->d_egdes_ids[index];
      #else
	return this->egdes_ids[index];
      #endif
}

/**
 * \brief Get the array of a node variable poiters on GPU.
 * \param var the id of variable array.
 * \return the array of variables poiters.
 */
__host__
inline T* get_variables_node(int var)
{
	return this->var_nodes_d_pointes[var];
}

/**
 * \brief Set the value of a source offset.
 * \param index the index on source offset array.
 * \param value the new value.
 */
__host__ __device__
inline void set_source_offsets(int index, int value)
{
      #ifdef __CUDA_ARCH__
	this->d_source_offsets[index]=value;
      #else
	this->source_offsets[index]=value;
      #endif
}

/**
 * \brief Get the value of a source offset.
 * \param index the index on source offset array.
 * \return the value.
 */
__host__ __device__
inline int get_source_offset(int index)
{
      #ifdef __CUDA_ARCH__
	return this->d_source_offsets[index];
      #else
	return this->source_offsets[index];
      #endif
}

/**
 * \brief Get the value of a destination indice.
 * \param index the index on destination indice array.
 * \return the value.
 */
__host__ __device__
inline int get_destination_indice(int index)
{
      #ifdef __CUDA_ARCH__
	return this->d_destination_indices[index];
      #else
	return this->destination_indices[index];
      #endif
}

/**
 * \brief Set the value of a destination indice.
 * \param index the index on destination indice array.
 * \param value the new value.
 */
__host__ __device__
inline void set_destination_indices(int index, int value)
{
      #ifdef __CUDA_ARCH__
	this->d_destination_indices[index]=value;
      #else
	this->destination_indices[index]=value;
      #endif
}

/**
 * \brief Get the array of source offset.
 * \return array of source offset.
 */
__host__ __device__
inline int* get_source_offsets()
{
      #ifdef __CUDA_ARCH__
	return this->d_source_offsets;
      #else
	return this->source_offsets;
      #endif
}

/**
 * \brief Get the array of destination indice.
 * \return array of destination indice.
 */
__host__ __device__
inline int* get_destination_indices()
{
      #ifdef __CUDA_ARCH__
	return this->d_destination_indices;
      #else
	return this->destination_indices;
      #endif
}

/**
 * \brief Get the array of a node variable poiters.
 * \param var the id of variable array.
 * \return the array of variables poiters.
 */
__host__ __device__
inline T* get_var_edges(int var)
{
      #ifdef __CUDA_ARCH__
	return this->d_var_edges[var];
      #else
	return this->var_edges[var];
      #endif
}

/**
 * \brief Get the array of a edge variable poiters.
 * \param var the id of variable array.
 * \return the array of variables poiters.
 */
__host__ __device__
inline T* get_var_nodes(int var)
{
      #ifdef __CUDA_ARCH__
	return this->d_var_nodes[var];
      #else
	return this->var_nodes[var];
      #endif
}


/**
 * \brief Get the current graph data memory location.
 * \return The graph data location state.
 */
__host__
inline graph_state get_state()
{
	return this->state;
}

__host__
inline void cpu_modified()
{
	this->state = GRAPH_ON_CPU;
}

__host__
inline int* get_all_node_type(){
	return this->node_type;
}

__host__ __device__
inline int get_node_type(int node_id)
{
      #ifdef __CUDA_ARCH__
	return this->d_node_type[node_id];
      #else
	return this->node_type[node_id];
      #endif
}

__host__ __device__
inline int* get_node_type_ptr(int node_id)
{
      #ifdef __CUDA_ARCH__
	return &this->d_node_type[node_id];
      #else
	return &this->node_type[node_id];
      #endif
}

__host__ __device__
inline void set_node_type(int node_id, int type)
{
      #ifdef __CUDA_ARCH__
	this->d_node_type[node_id] = type;
      #else
	this->node_type[node_id] = type;
      #endif
}

__host__ __device__
inline int get_group_id(int node_id)
{
      #ifdef __CUDA_ARCH__
	return this->d_group_id[node_id];
      #else
	return this->group_id[node_id];
      #endif
}


__host__ __device__
inline void set_group_id(int node_id, int id)
{
      #ifdef __CUDA_ARCH__
	this->d_group_id[node_id] = id;
      #else
	this->group_id[node_id] = id;
      #endif
}

__host__ __device__
inline int get_allocation_to_nodes_ids(int node_id)
{
      #ifdef __CUDA_ARCH__
	return this->d_allocation_to_nodes_ids[node_id];
      #else
	return this->allocation_to_nodes_ids[node_id];
      #endif
}

__host__ __device__
inline void set_allocation_to_nodes_ids(int node_id, int id)
{
      #ifdef __CUDA_ARCH__
	this->d_allocation_to_nodes_ids[node_id] = id;
      #else
	this->allocation_to_nodes_ids[node_id] = id;
      #endif
}

__host__
inline std::vector<int>* get_allocation_to_edges_ids(int edge_id)
{
	return this->allocation_to_edges_ids[this->egdes_ids[edge_id]];

}

__host__
inline void add_allocation_to_edges_ids(int edge_id, int id)
{
	this->allocation_to_edges_ids[this->egdes_ids[edge_id]]->push_back(id);
}

__host__
inline void set_allocation_to_edges_ids(int edge_id, std::vector<int>* vec)
{
	this->allocation_to_edges_ids[this->egdes_ids[edge_id]] = vec;
}

__host__
inline bool get_is_grouped(){
	return this->is_grouped;
}

__host__
inline void set_group_graph(graph* g){
	this->group_graph = g;
}

__host__
inline graph* get_group_graph(){
	return this->group_graph;
}

__host__
inline void set_is_grouped(bool grouped){
	this->is_grouped = grouped;
}

__host__
inline bool get_is_allocated(){
	return this->is_allocated;
}

__host__
inline int* get_group_d_ptr(){
	return this->d_group_id;
}

__host__
inline int* get_group_ptr(){
	return this->group_id;
}

__host__
inline void set_is_allocated(bool allocated){
	this->is_allocated = allocated;
}


__host__
inline void add_node_variable_name(std::string name)
{
	this->nodes_variables_names->push_back(name);
}

__host__
inline void add_edge_variable_name(std::string name)
{
	this->edges_variables_names->push_back(name);
}


__host__
void check_edges_ids(){

	int number_edges = 0;
	for(int id=0; id<this->get_num_nodes(); id++)
	{
		for(int i=this->get_source_offset(id); i<this->get_source_offset(id+1); i++)
		{
			int neighbor = this->get_destination_indice(i);
			if(neighbor>id) {
				this->set_egdes_ids(i,number_edges);
				number_edges++;
			}else{
				for(int y=this->get_source_offset(neighbor); y<this->get_source_offset(neighbor+1); y++)
				{
					int neighbor_2 = this->get_destination_indice(y);
					if(id==neighbor_2) {
						this->set_egdes_ids(i, this->get_egdes_ids(y));
						break;
					}
				}
			}
		}
	}



}

//check if already grouped
void initialize_group(){
	if(!this->is_grouped) {
		this->is_grouped = true;
		cudaMallocHost((void**)&this->group_id,sizeof(int)*this->num_nodes);
		cudaMalloc(&this->d_group_id, sizeof(int)*this->num_nodes);
	}
}

//check if already allocated
void initialize_allocation(){
	if(!this->is_allocated) {
		this->set_is_allocated(true);
		cudaMallocHost((void**)&this->allocation_to_nodes_ids,sizeof(int)*this->num_nodes);
		cudaMalloc(&this->d_allocation_to_nodes_ids, sizeof(int)*this->num_nodes);
		this->allocation_to_edges_ids =  (std::vector<int>**)malloc(sizeof(std::vector<int>*)*this->num_edges);
		for(int i=0; i<this->num_edges; i++) {
			this->allocation_to_edges_ids[i] = new std::vector<int>();
		}
	}
}

/**
 * \brief Update the graph data on Device coping the data on Host.
 */
void update_gpu(bool force=false)
{

	if(this->state==GRAPH_ON_GPU && !force) {
		return;
	}

	cudaMemcpy(this->d_source_offsets, this->source_offsets, sizeof(int)*(this->num_nodes+1), cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_destination_indices, this->destination_indices, sizeof(int)*this->num_edges*2, cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_egdes_ids, this->egdes_ids, sizeof(int)*this->num_edges*2, cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_node_type, this->node_type, sizeof(int)*this->num_nodes, cudaMemcpyHostToDevice);

	if(this->is_grouped)
	{
		cudaMemcpy(this->d_group_id, this->group_id, sizeof(int)*this->num_nodes, cudaMemcpyHostToDevice);
	}

	if(this->is_allocated)
	{
		cudaMemcpy(this->d_allocation_to_nodes_ids, this->allocation_to_nodes_ids, sizeof(int)*this->num_nodes, cudaMemcpyHostToDevice);
		//cudaMemcpy(this->d_allocation_to_edges_ids, this->allocation_to_edges_ids, sizeof(int)*this->num_edges, cudaMemcpyHostToDevice);
	}

	for(int i=0; i<this->num_var_nodes; i++)
	{
		cudaMemcpy(this->var_nodes_d_pointes[i], this->var_nodes[i], sizeof(T)*this->num_nodes, cudaMemcpyHostToDevice);
	}

	for(int i=0; i<this->num_var_edges; i++)
	{
		cudaMemcpy(this->var_edges_d_pointes[i], this->var_edges[i], sizeof(T)*this->num_edges, cudaMemcpyHostToDevice);
	}

	state=GRAPH_ON_GPU;

}

/**
 * \brief Update the graph variables data on Device coping the data on Host.
 */
void update_variables_gpu()
{

	if(this->state==GRAPH_ON_GPU) {
		throw std::runtime_error("Graph is not on GPU.");
	}

	for(int i=0; i<this->num_var_nodes; i++)
	{
		cudaMemcpy(this->var_nodes_d_pointes[i], this->var_nodes[i], sizeof(T)*this->num_nodes, cudaMemcpyHostToDevice);
	}

	for(int i=0; i<this->num_var_edges; i++)
	{
		cudaMemcpy(this->var_edges_d_pointes[i], this->var_edges[i], sizeof(T)*this->num_edges, cudaMemcpyHostToDevice);
	}

	state=GRAPH_ON_GPU;
}

/**
 * \brief Update the graph data on Host coping the data on Device.
 */
void update_cpu(bool force=false)
{

	if(this->state==GRAPH_ON_CPU && !force) {
		return;
	}

	cudaMemcpy(this->source_offsets, this->d_source_offsets, sizeof(int)*(this->num_nodes+1), cudaMemcpyDeviceToHost);
	cudaMemcpy(this->destination_indices, this->d_destination_indices, sizeof(int)*this->num_edges*2, cudaMemcpyDeviceToHost);
	cudaMemcpy(this->egdes_ids, this->d_egdes_ids, sizeof(int)*this->num_edges*2, cudaMemcpyDeviceToHost);
	cudaMemcpy(this->node_type, this->d_node_type, sizeof(int)*this->num_nodes, cudaMemcpyDeviceToHost);

	if(this->is_grouped)
	{
		cudaMemcpy(this->group_id, this->d_group_id, sizeof(int)*this->num_nodes, cudaMemcpyDeviceToHost);
	}

	if(this->is_allocated)
	{
		cudaMemcpy(this->allocation_to_nodes_ids, this->d_allocation_to_nodes_ids, sizeof(int)*this->num_nodes, cudaMemcpyDeviceToHost);
		//cudaMemcpy(this->allocation_to_edges_ids, this->d_allocation_to_edges_ids, sizeof(int)*this->num_edges, cudaMemcpyDeviceToHost);
	}


	for(int i=0; i<this->num_var_nodes; i++)
	{
		cudaMemcpy(this->var_nodes[i], this->var_nodes_d_pointes[i], sizeof(T)*this->num_nodes, cudaMemcpyDeviceToHost);
	}

	for(int i=0; i<this->num_var_edges; i++)
	{
		cudaMemcpy(this->var_edges[i], this->var_edges_d_pointes[i], sizeof(T)*this->num_edges, cudaMemcpyDeviceToHost);
	}

	state=GRAPH_ON_CPU;

}

/**
 * \brief Update the graph variables data on Host coping the data on Device.
 */
void update_variables_cpu()
{

	if(this->state==GRAPH_ON_CPU) {
		throw std::runtime_error("Graph is not on GPU.");
	}

	for(int i=0; i<this->num_var_nodes; i++)
	{
		cudaMemcpy(this->var_nodes[i], this->var_nodes_d_pointes[i], sizeof(T)*this->num_nodes, cudaMemcpyDeviceToHost);
	}

	for(int i=0; i<this->num_var_edges; i++)
	{
		cudaMemcpy(this->var_edges[i], this->var_edges_d_pointes[i], sizeof(T)*this->num_edges, cudaMemcpyDeviceToHost);
	}

	state=GRAPH_ON_CPU;
}

/**
 * \brief Add a new node variable allocating all the memory needed.
 * \return The new node variable ID.
 */
int add_node_variable()
{
	std::string name = std::string("Var ")+std::to_string(this->num_var_nodes);
	return add_node_variable(name);
}

/**
 * \brief Add a new node variable allocating all the memory needed.
 * \return The new node variable ID.
 */
int add_node_variable(std::string var_name)
{
	if(this->num_var_nodes >= GRAPH_MAX_VARIABLES)
	{
		throw std::runtime_error("Max variables.");
	}

	int new_index = this->num_var_nodes++;

	cudaMallocHost((void**)&this->var_nodes[new_index],sizeof(T)*this->num_nodes);
	cudaMalloc(&this->var_nodes_d_pointes[new_index], sizeof(T)*this->num_nodes);
	cudaMemcpy(this->d_var_nodes, this->var_nodes_d_pointes, sizeof(T*)*this->num_var_nodes, cudaMemcpyHostToDevice);

	this->add_node_variable_name(var_name);

	return new_index;
}

int add_edge_variable()
{
	std::string name = std::string("Var ")+std::to_string(this->num_var_edges);
	return add_node_variable(name);
}

/**
 * \brief Add a new edge variable allocating all the memory needed.
 * \return The new edge variable ID.
 */
int add_edge_variable(std::string var_name)
{
	if(this->num_var_edges >= GRAPH_MAX_VARIABLES)
	{
		throw std::runtime_error("Max variables.");
	}

	int new_index = this->num_var_edges++;

	cudaMallocHost((void**)&this->var_edges[new_index],sizeof(T)*this->num_edges);
	cudaMalloc(&this->var_edges_d_pointes[new_index], sizeof(T)*this->num_edges);
	cudaMemcpy(this->d_var_edges, this->var_edges_d_pointes, sizeof(T*)*this->num_var_edges, cudaMemcpyHostToDevice);

	this->add_edge_variable_name(var_name);

	return new_index;
}



//Only compile this block if there is a XML lib.
    #ifndef NO_XML_LIB
/**
 * \brief Save the graph on the GEXF format.
 * \param path the path to new file.
 */
void save_to_gexf(const char* path)
{
	pugi::xml_document doc;
	pugi::xml_node gexf = doc.append_child("gexf");
	gexf.append_attribute("xmlns") = "http://www.gexf.net/1.3";
	gexf.append_attribute("version") = "1.3";

	gexf.append_attribute("xmlns:viz") = "http://www.gexf.net/1.3/viz";
	gexf.append_attribute("xmlns:xsi") = "http://www.w3.org/2001/XMLSchema−instance";
	gexf.append_attribute("xsi:schemaLocation") = "http://www.gexf.net/1.3 http://www.gexf.net/1.3/gexf.xsd";


	pugi::xml_node meta = gexf.append_child("meta");
	meta.append_attribute("lastmodifieddate") = "2017-01-01";

	pugi::xml_node creator = meta.append_child("creator");
	creator.append_child(pugi::node_pcdata).set_value("Vne Cuda");

	pugi::xml_node description = meta.append_child("description");
	description.append_child(pugi::node_pcdata).set_value("-");

	pugi::xml_node _xmlGraph = gexf.append_child("Graph");
	_xmlGraph.append_attribute("defaultedgetype") = "undirected";
	_xmlGraph.append_attribute("mode") = "static";

	int group_var_id;
	int allocation_node_var_id;

	if(this->num_var_nodes>0 || this->is_grouped || this->is_allocated) {
		pugi::xml_node attributes_node = _xmlGraph.append_child("attributes");
		attributes_node.append_attribute("class") = "node";
		attributes_node.append_attribute("mode") = "static";
		int var;
		for(var=0; var<this->num_var_nodes; var++) {
			pugi::xml_node atribute = attributes_node.append_child("attribute");
			atribute.append_attribute("id")=var;
			atribute.append_attribute("title")=nodes_variables_names->at(var).c_str();
			atribute.append_attribute("type")="float";
		}

		if(this->is_grouped) {
			pugi::xml_node atribute = attributes_node.append_child("attribute");
			group_var_id=var++;
			atribute.append_attribute("id")=group_var_id;
			atribute.append_attribute("title")="Group";
			atribute.append_attribute("type")="integer";
		}

		if(this->is_allocated) {
			pugi::xml_node atribute = attributes_node.append_child("attribute");
			allocation_node_var_id=var++;
			atribute.append_attribute("id")=allocation_node_var_id;
			atribute.append_attribute("title")="Allocation Id";
			atribute.append_attribute("type")="integer";
		}

	}

	if(this->num_var_edges>0)
	{
		pugi::xml_node attributes_edge = _xmlGraph.append_child("attributes");
		attributes_edge.append_attribute("class") = "edge";
		attributes_edge.append_attribute("mode") = "static";

		for(int var=0; var<this->num_var_edges; var++)
		{
			pugi::xml_node atribute = attributes_edge.append_child("attribute");
			atribute.append_attribute("id")=var;
			atribute.append_attribute("title")=edges_variables_names->at(var).c_str();
			atribute.append_attribute("type")="float";
		}

	}


	pugi::xml_node nodes = _xmlGraph.append_child("nodes");

	for(int i=0; i<this->num_nodes; i++)
	{
		pugi::xml_node node = nodes.append_child("node");
		node.append_attribute("id") = i;
		node.append_attribute("label") = node_type[i];

		if(this->num_var_nodes>0 || this->is_grouped || this->is_allocated)
		{
			pugi::xml_node attributes_node = node.append_child("attvalues");

			for(int var=0; var<this->num_var_nodes; var++)
			{
				pugi::xml_node atribute = attributes_node.append_child("attvalue");
				atribute.append_attribute("for")=var;
				atribute.append_attribute("value")=(T)this->var_nodes[var][i];
			}

			if(this->is_grouped) {
				pugi::xml_node atribute = attributes_node.append_child("attvalue");
				atribute.append_attribute("for")=group_var_id;
				atribute.append_attribute("value")=this->group_id[i];
			}

			if(this->is_allocated) {
				pugi::xml_node atribute = attributes_node.append_child("attvalue");
				atribute.append_attribute("for")=allocation_node_var_id;
				atribute.append_attribute("value")=this->allocation_to_nodes_ids[i];
			}

		}
	}


	pugi::xml_node edges = _xmlGraph.append_child("edges");

	for(int i=0; i<this->num_nodes; i++)
	{
		for(int end=this->source_offsets[i]; end<this->source_offsets[i+1]; end++)
		{
			pugi::xml_node edge = edges.append_child("edge");
			edge.append_attribute("id") = end;
			edge.append_attribute("source") = i;
			edge.append_attribute("target") = this->destination_indices[end];

			if(this->num_var_edges>0)
			{
				pugi::xml_node attributes_node = edge.append_child("attvalues");

				for(int var=0; var<this->num_var_edges; var++)
				{
					pugi::xml_node atribute = attributes_node.append_child("attvalue");
					atribute.append_attribute("for")=var;
					atribute.append_attribute("value")=this->get_variable_edge(var, end);
				}

			}
		}
	}

	doc.save_file(path);
}
    #endif


__host__ __device__
~graph()
{
	//TODO: Maybe Host Free all and Device do nothing?
}

__host__
void free_graph()
{

	cudaFreeHost(source_offsets);
	cudaFreeHost(destination_indices);
	cudaFreeHost(egdes_ids);
	CUDA_CHECK();
	for(int i=0; i<num_var_nodes; i++) {
		cudaFreeHost(var_nodes[i]);
		cudaFree(var_nodes_d_pointes[i]);
	}

	cudaFreeHost(var_nodes);
	cudaFreeHost(var_nodes_d_pointes);

	for(int i=0; i<num_var_edges; i++) {
		cudaFreeHost(var_edges[i]);
		cudaFree(var_edges_d_pointes[i]);
	}
	CUDA_CHECK();
	cudaFreeHost(var_edges);
	cudaFreeHost(var_edges_d_pointes);

	cudaFree(d_source_offsets);
	cudaFree(d_destination_indices);
	cudaFree(d_egdes_ids);
	cudaFree(d_var_nodes);
	cudaFree(d_var_edges);

	delete nodes_variables_names;
	delete edges_variables_names;

	cudaFreeHost(node_type);

	cudaFree(d_node_type);

	if(is_grouped) {
		cudaFreeHost(group_id);
		cudaFree(d_group_id);
		//group_graph->free_graph();
	}
	CUDA_CHECK();
	if(distance_matrix!=NULL) {
		distance_matrix->free();
	}

	CUDA_CHECK();

	if(is_allocated) {
		cudaFreeHost(allocation_to_nodes_ids);
		cudaFree(d_allocation_to_nodes_ids);

		for(int i=0; i<this->num_edges; i++) {
			delete allocation_to_edges_ids[i];
		}
		free(allocation_to_edges_ids);
	}

	CUDA_CHECK();

}




private:
//General info.
unsigned int num_nodes;       ///< Number of nodes.
unsigned int num_edges;       ///< Number of edges.
unsigned int num_var_nodes;       ///< Number of nodes variables.
unsigned int num_var_edges;       ///< Number of edges variables.
unsigned int hosts;
graph_state state;      ///< State where the data was last modificated.

//Host Topology info.
int *source_offsets;       ///< Offsets for CSR on Host.
int *destination_indices;       ///< Destinations for CSR on Host.
int *egdes_ids;
T **var_nodes;       ///< nodes variables on Host.
T **var_edges;       ///< Edges variables on Host.

//Device Topology info.
int* d_source_offsets;       ///< Offsets for CSR on Device.
int* d_destination_indices;       ///< Destinations for CSR on Device.
int* d_egdes_ids;
T **d_var_nodes;       ///< nodes variables on Device.
T **d_var_edges;       ///< Edges variables on Device.

//Device Topology reference on Host.
T **var_nodes_d_pointes;       ///< Host array with Device pointes for nodes variables.
T **var_edges_d_pointes;       ///< Host array with Device pointes for edges variables.

std::vector<std::string>* nodes_variables_names;
std::vector<std::string>* edges_variables_names;

int* node_type;
int* d_node_type;

//Distance Info
vnegpu::util::matrix<T>* distance_matrix=NULL;

//Group Info
bool is_grouped=false;
graph* group_graph;
int* group_id;
int* d_group_id;


//Allocation Info
bool is_allocated=false;
graph* host_graph;
int* allocation_to_nodes_ids;
std::vector<int>** allocation_to_edges_ids;
int* d_allocation_to_nodes_ids;



};   // end graph


} // end vnegpu

#endif
