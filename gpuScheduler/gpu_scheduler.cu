#include <iostream>
#include <chrono>
#include <ctime>
#include <ratio>

#include "builder.cuh"
#include "thirdparty/clara.hpp"

#include "allocator/naive.hpp"
#include "allocator/free.hpp"
#include "allocator/all.cuh"
#include "allocator/dc.cuh"

namespace Allocation_t {
enum {
	NAIVE, DC, ALL
};
}

typedef struct {
	int allocation_type=Allocation_t::NAIVE;
	std::string multicriteria_method;
	std::string clustering_method;
	std::string topology_type;
	int topology_size=0;
	int current_time=0;
	int start_time=0;
	int test_type=0;
	int end_time=0;
} options_t;

typedef struct {
	std::map<int, const char*> allocated_task;
	std::vector<Container*> containers;
	int total_containers=0;
	int total_accepted=0;
	int total_refused=0;
} scheduler_t;

void setup(int argc, char** argv, Builder* builder, scheduler_t *scheduler, options_t* options){
	std::string topology = "fat_tree";
	std::string multicriteria_method = "ahpg";
	std::string clustering_method = "mcl";
	std::string allocation_type = "naive";
	int topology_size=10;
	int start_scheduler_time=0;
	int end_time=-1;
	// dont show the help by default. Use `-h or `--help` to enable it.
	bool showHelp = false;
	int test_type=0;
	auto cli = clara::detail::Help(showHelp)
	           | clara::detail::Opt( topology, "topology" )["-t"]["--topology"]("What is the topology type? [ (default) fat_tree | dcell | bcube ]")
	           | clara::detail::Opt( topology_size, "topology size") ["-s"] ["--topology_size"] ("What is the size of the topology? ( default 10 )")
	           | clara::detail::Opt( multicriteria_method, "multicriteria method") ["-m"]["--multicriteria"] ("What is the multicriteria method? [ ahp | (default) ahpg ]")
	           | clara::detail::Opt( clustering_method,"clustering method") ["-c"]["--clustering"] ("What is the clustering method? [ (default) mcl ]")
	           | clara::detail::Opt( allocation_type,"Strategy") ["-a"] ["--strategy"] ("What is the allocation strategy? [ (default) naive | dc |  all ]")
	           | clara::detail::Opt( start_scheduler_time, "Start Time")["--start_time"]("Start scheduler time")
	           | clara::detail::Opt( end_time, "Finish Time")["--end_time"]("What is the schaduler max time?")
	           // 0 For no Test
	           // 1 For Container Test
	           | clara::detail::Opt( test_type, "Type Test")["--test"]("Which type of test you want?");
	auto result = cli.parse( clara::detail::Args( argc, argv ) );
	if( !result ) {
		std::cerr << "Error in command line: " << result.errorMessage() <<std::endl;
		exit(1);
	}
	if ( showHelp ) {
		std::cout << cli << std::endl;
		exit(0);
	}
	if( topology != "fat_tree" && topology != "dcell" && topology != "bcube" ) {
		std::cerr << "Invalid entered topology\n";
		exit(0);
	}else{
		options->topology_type=topology;
	}
	if( (topology_size<2 || topology_size>48) && topology_size!=0) {
		std::cerr << "Invalid topology size ( must be between 4 and 48 )\n";
		exit(0);
	}else{
		options->topology_size=topology_size;
	}
	if( multicriteria_method == "ahpg") {
		builder->setAHPG();
		options->multicriteria_method=multicriteria_method;
	}else if(multicriteria_method == "ahp" ) {
		builder->setAHP();
		options->multicriteria_method=multicriteria_method;
	}else{
		std::cerr << "Invalid multicriteria method\n";
		exit(0);
	}
	if( clustering_method == "mcl" ) {
		builder->setMCL();
		options->clustering_method=clustering_method;
	}else{
		std::cerr << "Invalid clustering method\n";
		exit(0);
	}
	if( start_scheduler_time < 0 ) {
		std::cerr << "Invalid start scheduler time\n";
		exit(0);
	}else{
		options->start_time=start_scheduler_time;
	}
	if( end_time< -1 ) {
		std::cerr << "Invalid end scheduler time\n";
		exit(0);
	}else{
		options->end_time=end_time;
	}
	if( allocation_type=="naive") {
		options->allocation_type=Allocation_t::NAIVE;
	}else if(allocation_type=="dc") {
		options->allocation_type=Allocation_t::DC;
	}else if(allocation_type=="all") {
		options->allocation_type=Allocation_t::ALL;
	}else{
		std::cerr << "Invalid allocation type\n";
		exit(0);
	}
	if(test_type >=0 && test_type<2) {
		options->test_type=test_type;
	}else{
		std::cerr << "Invalid Type of test\n" << test_type << "\n";
		exit(0);
	}
	options->current_time=options->start_time;
	// Load the Topology
	std::string path="datacenter/json/"+topology+"/"+std::to_string(topology_size)+".json";
	builder->parser(path.c_str());
}

inline void update_scheduler(scheduler_t* scheduler, bool allocation_success){
	scheduler->total_containers++;
	allocation_success == true ? scheduler->total_accepted++ : scheduler->total_refused++;
	if(scheduler->total_refused+scheduler->total_accepted!=scheduler->total_containers) {
		std::cerr << "Erro in containers total check\n";
		exit(1);
	}
}

inline void delete_tasks(scheduler_t* scheduler, Builder* builder, options_t* options){
	bool free_success=false;
	// printf("%s #### %s\n", scheduler->allocated_task[0].c_str(),scheduler->allocated_task[1].c_str());

	for(std::vector<Container*>::iterator it=scheduler->containers.begin(); it!=scheduler->containers.end();) {
		// std::cout<<"new container test\n";
		if((*it)->getDuration()+(*it)->getSubmission()==options->current_time) {
			// std::cout<<"######################################\n";
			// std::cout << " Delete Task QUEUE SIZE " << scheduler->containers.size()<<" \n";

			// std::cout << "Container ID: "<< (*it)->getId() << " has duration of "<< (*it)->getDuration() << " and is submitted in " << (*it)->getSubmission()<<"\n";
			// Before remove the container, free the consumed space in the Data Center
			// std::cout<<"Task allocated with id " << (*it)->getId() << " has name " << scheduler->allocated_task[(*it)->getId()]<<" .\n";
			// std::cout<<"Task allocated with id 1 " << scheduler->allocated_task[1]<<" .\n";
			// std::cout<<"\n";
			// if ( scheduler->allocated_task.find(0) == scheduler->allocated_task.end() ) {
			// std::cout<<" DELETADO ESSA POCILGA\n";
			// } else {
			// std::cout<<" POCILGA NAO DELETADO\n";
			// }
			free_success=Allocator::freeHostResource(
				/* the specific host that have the container*/
				builder->getHost(scheduler->allocated_task[(*it)->getId()]),
				/* The container to be removed*/
				(*it)
				);
			if(!free_success) {
				std::cerr << "gpu_scheduler(170) - Error in free the task " << (*it)->getId() << " from the data center\n";
				exit(1);
			}
			// Search the container C in the vector and removes it
			scheduler->allocated_task.erase((*it)->getId());

			scheduler->containers.erase(std::remove(scheduler->containers.begin(), scheduler->containers.end(), (*it)), scheduler->containers.end());
			// std::cout<<"######################################\n";
			// std::cout << "Deleted Complete\n";
		} else it++;
	}

}


inline void allocate_tasks(scheduler_t* scheduler, Builder* builder, options_t* options){
	// std::cout << "Try to allocate\n";
	bool allocation_success=false;
	// Check the task submission
	for(Container *c : scheduler->containers) {
		if( options->current_time == (int)c->getSubmission()) {
			// allocate the new task in the data center.
			// std::cout<<"Allocating\n";
			if(options->allocation_type==Allocation_t::NAIVE) {
				// std::cout<<"Naive\n";
				allocation_success=Allocator::naive(builder,c, scheduler->allocated_task);
				// std::cout<<"Allocated\n";
			}else if(options->allocation_type==Allocation_t::DC) {
				allocation_success=Allocator::dc(builder,c,scheduler->allocated_task);
			}else if(options->allocation_type==Allocation_t::ALL) {
				allocation_success=Allocator::all();
			}
			else{
				std::cerr << "Invalid type\n";
			}
			if(!allocation_success) {
				std::cerr << "Error in allocate\n";
				exit(3);
			}else{
				// std::cout << "ALLOCATED! "<<scheduler->allocated_task[c->getId()]<<"##\n";
			}
			// getchar();
		}
	}
	// update_scheduler(scheduler, allocation_success);
}

void schedule(Builder* builder, Comunicator* conn, scheduler_t* scheduler, options_t* options, int message_count){
	// if(options->end_time==-1) {
	// #define NO_END_TIME
	// }
	// #ifdef NO_END_TIME
	// while(message_count>0) {
	// #else
	while(message_count>0 || options->current_time <= options->end_time) {
		// #endif
		// if(options->current_time==options->end_time) break;
		// std::cout<<"Scheduler Time "<< options->current_time<<"\n";
		// std::cout<<"message_count "<<message_count<<"\n";
		// std::cout<<"contianers size "<<scheduler->containers.size()<<"\n";
		// make sure there is work in the queue
		if(message_count>0) {
			// Create new container
			Container *c = new Container();
			// Set the resources to the container
			c->setTask(conn->getNextTask());
			// Put the container in the vector
			scheduler->containers.push_back(c);
			message_count--;
			// Print the container
			// std::cout << *c << "\n";
		}
		// Search the containers to delete
		// printf("Delete\n");
		delete_tasks(scheduler, builder, options);
		// Search the containers in the vector to allocate in the DC
		// std::cout<<"New Allocation\n";
		// printf("Allocate\n");
		allocate_tasks(scheduler, builder, options);
		// std::cout<<"Done Allocation\n";
		// Update the lifetime
		options->current_time++;
		// printf(" Checked\n");
		// getchar();
	}
}

int main(int argc, char **argv){
	// Options Struct
	options_t* options = new options_t;
	// Scheduler Struct
	scheduler_t* scheduler = new scheduler_t;
	// Creating the communicatior
	Comunicator *conn = new Comunicator();
	conn->setup();
	int message_count=conn->getQueueSize();
	// Create the Builder
	Builder *builder= new Builder();
	// Parse the command line arguments
	setup(argc,argv,builder,scheduler, options);

	// std::cout<<"Multicriteria method;Fat Tree Size;Number of containers;Time\n";

	if (options->test_type==0) {         // no test is set
		schedule(builder, conn, scheduler, options, message_count);
	}else if(options->test_type==1) {        // container test
		// force cout to not print in cientific notation
		options->end_time = message_count+2;
		std::cout<<std::fixed;

		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		schedule(builder, conn, scheduler, options, message_count);

		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
		std::cout<<options->multicriteria_method<<";"<<options->topology_size<<";"<<message_count<<";"<<time_span.count()<<"\n";
	}
	// Free the allocated pointers
	delete(scheduler);
	delete(builder);
	delete(options);
	delete(conn);
	return 0;
}
