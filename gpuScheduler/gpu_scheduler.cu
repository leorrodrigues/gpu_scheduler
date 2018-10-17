#include <iostream>
#include <chrono>
#include <ctime>
#include <ratio>

#include "builder.cuh"
#include "thirdparty/clara.hpp"

#include "allocator/naive.hpp"
#include "allocator/data.hpp"
#include "allocator/free.hpp"
#include "allocator/all.hpp"

namespace Allocation_t {
enum {
	NAIVE, DATA, ALL
};
}

typedef struct {
	int time=6580;
	int total_containers=0;
	int total_refused=0;
	int total_accepted=0;
	std::vector<Container*> containers;
	std::map<int,std::string> allocated_task;
	int allocation_type=Allocation_t::NAIVE;
} scheduler_t;

void setup(int argc, char** argv, Builder* builder, scheduler_t *scheduler){
	std::string topology = "fat_tree";
	std::string multicriteria_method = "ahpg";
	std::string clustering_method = "mcl";
	std::string type = "naive";
	int topology_size=10;

	// dont show the help by default. Use `-h or `--help` to enable it.
	bool showHelp = false;
	auto cli = clara::detail::Help(showHelp)
	           | clara::detail::Opt( topology, "topology" )["-t"]["--topology"]("What is the topology type? [ (default) fat_tree | dcell | bcube ]")
	           | clara::detail::Opt( topology_size, "topology size") ["-s"] ["--topology_size"] ("What is the size of the topology? ( default 10 )")
	           | clara::detail::Opt( multicriteria_method, "multicriteria method") ["-m"]["--multicriteria"] ("What is the multicriteria method? [ ahp | (default) ahpg ]")
	           | clara::detail::Opt( clustering_method,"clustering method") ["-c"]["--clustering"] ("What is the clustering method? [ (default) mcl ]")
	           | clara::detail::Opt(  type,"Strategy") ["-a"] ["--strategy"] ("What is the allocation strategy? [ (default) naive | data |  all ]");
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
	}
	if( topology_size<2 || topology_size>48) {
		std::cerr << "Invalid topology size ( must be between 4 and 48 )\n";
		exit(0);
	}
	if( multicriteria_method == "ahpg") {
		builder->setAHPG();
	}else if(multicriteria_method == "ahp" ) {
		builder->setAHP();
	}else{
		std::cerr << "Invalid multicriteria method\n";
		exit(0);
	}
	if( clustering_method == "mcl" ) {
		builder->setMCL();
	}else{
		std::cerr << "Invalid clustering method\n";
		exit(0);
	}
	if( type=="naive") {
		scheduler->allocation_type=Allocation_t::NAIVE;
	}else if(type=="data") {
		scheduler->allocation_type=Allocation_t::DATA;
	}else if(type=="all") {
		scheduler->allocation_type=Allocation_t::ALL;
	}else{
		std::cerr << "Invalid allocation type\n";
		exit(0);
	}
	// Load the Topology
	std::string path="datacenter/json/"+topology+"/"+std::to_string(topology_size)+".json";
	builder->parser(path.c_str());
}

inline void update_lifetime(scheduler_t* scheduler){
	// for(Container *c : scheduler->containers) {
	for(Container* c : scheduler->containers) {
		c->decreaseDuration(1);
	}
}

inline void update_scheduler(scheduler_t* scheduler, bool allocation_success){
	scheduler->time++;
	scheduler->total_containers++;
	allocation_success == true ? scheduler->total_accepted++ : scheduler->total_refused++;
	if(scheduler->total_refused+scheduler->total_accepted!=scheduler->total_containers) {
		std::cerr << "Erro in containers total check\n";
		exit(1);
	}
}

inline void delete_tasks(scheduler_t* scheduler, Builder* builder){
	bool free_success=false;
	std::cout<<"######################################\n";

	std::cout << " Delete Task QUEUE SIZE " << scheduler->containers.size()<<" \n";
	for(std::vector<Container*>::iterator it=scheduler->containers.begin(); it!=scheduler->containers.end();) {
		// std::cout<<"new container test\n";
		if((*it)->getDuration()+(*it)->getSubmission()==scheduler->time) {

			std::cout << "Container "<< (*it)->getId() << " = "<< (*it)->getDuration() << " " << (*it)->getSubmission()<<"\n";
			// Before remove the container, free the consumed space in the Data Center
			free_success=Allocator::freeHostResource(
				/* the specific host that have the container*/
				builder->getHost(scheduler->allocated_task[(*it)->getId()]),
				/*The container to be removed*/
				(*it)
				);
			if(!free_success) {
				std::cerr << "Error in free the task " << (*it)->getId() << " from the data center\n";
				exit(1);
			}
			//Search the container C in the vector and removes it
			scheduler->containers.erase(std::remove(scheduler->containers.begin(), scheduler->containers.end(), (*it)), scheduler->containers.end());
			scheduler->allocated_task.erase((*it)->getId());
		} else it++;
	}
	std::cout<<"######################################\n";

	std::cout << "Deleted Complete\n";
}


inline void allocate_tasks(scheduler_t* scheduler, Builder* builder){
	std::cout << "Try to allocate\n";
	bool allocation_success=false;
	// Check the task submission
	for(Container *c : scheduler->containers) {
		if( scheduler->time == (int)c->getSubmission()) {
			// allocate the new task in the data center.
			if(scheduler->allocation_type==Allocation_t::NAIVE) {
				allocation_success=Allocator::naive(builder,c, scheduler->allocated_task);
			}else if(scheduler->allocation_type==Allocation_t::DATA) {
				allocation_success=Allocator::data();
			}else if(scheduler->allocation_type==Allocation_t::ALL) {
				allocation_success=Allocator::all();
			}
			else{
				std::cerr << "Invalid type\n";
			}
			if(!allocation_success) {
				std::cerr << "Error in allocate\n";
				exit(3);
			}else{
				std::cerr << "ALLOCATED! "<<scheduler->allocated_task[0]<<"##\n";
			}
		}
	}
	update_scheduler(scheduler, allocation_success);
}

int main(int argc, char **argv){
	// Scheduler Struct
	scheduler_t* scheduler = new scheduler_t;
	// Creating the communicatior
	Comunicator *conn = new Comunicator();
	conn->setup();
	int message_count=conn->getQueueSize();
	// Create the Builder
	Builder *builder= new Builder();
	setup(argc,argv,builder,scheduler);

	while(message_count-- || !scheduler->containers.empty()) {
		fflush(stdin);
		std::cout<<"Scheduler Time "<< scheduler->time<<"\n";
		// make sure there is work in the queue
		if(message_count>0) {
			Container *c = new Container();
			c->setTask(conn->getNextTask());
			scheduler->containers.push_back(c);
			std::cout << *c << "\n";
		}
		// reduce all the tasks time by 1, when 0, removes it.
		printf("Deleting \n");
		delete_tasks(scheduler, builder);
		printf(" Checked\nAllocating\n");
		allocate_tasks(scheduler, builder);
		printf(" Checked\nLifetime\n");
		// update_lifetime(scheduler);
		printf(" Checked\n");
		getchar();
	}
	// if(builder->getHosts().size()>0) {
	//aI=std::chrono::steady_clock::now();
	//std::cout<<"Running MCL\n";
	//cI=std::chrono::steady_clock::now();
	// std::cout<<"Running clustering...\n";
	// builder->runClustering(builder->getHosts());

	//cF=std::chrono::steady_clock::now();
	//std::cout<<"Get MCL results\n";
	//crI=std::chrono::steady_clock::now();
	// std::cout<<"Getting clustering answer...\n";
	// builder->getClusteringResult();

	//crF=std::chrono::steady_clock::now();
	//std::cout<<"Running AHP\n";
	// mI=std::chrono::steady_clock::now();

	//builder->runMulticriteria( builder->getClusterHosts() );
	// builder->runMulticriteria(builder->getHosts());
	// mF=std::chrono::steady_clock::now();
	//std::cout<<"Getting AHP results\n";
	//builder->listCluster();
	//mrI=std::chrono::steady_clock::now();

	// auto results=builder->getMulcriteriaResult();
	return 0;
}
