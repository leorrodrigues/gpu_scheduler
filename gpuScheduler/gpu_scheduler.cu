#include <iostream>
#include <chrono>
#include <ctime>
#include <ratio>

#include "builder.cuh"
#include "thirdparty/clara.hpp"

namespace Allocation_Type {
enum {
	NAIVE, DATA, ALL
};
}

typedef struct {
	int time=0;
	int total_containers=0;
	int total_refused=0;
	int total_accepted=0;
	std::vector<Container*> containers;
	int type=Allocation_Type::NAIVE;
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
		scheduler->type=Allocation_Type::NAIVE;
	}else if(type=="data") {
		scheduler->type=Allocation_Type::DATA;
	}else if(type=="all") {
		scheduler->type=Allocation_Type::ALL;
	}else{
		std::cerr << "Invalid allocation type\n";
		exit(0);
	}
	// Load the Topology
	std::string path="datacenter/json/"+topology+"/"+std::to_string(topology_size)+".json";
	builder->parser(path.c_str());
}

inline void update_lifetime(scheduler_t* scheduler){
	std::for_each(scheduler->containers.begin(),  scheduler->containers.end(), [] (Container* c) mutable {
		printf("Before %d\n",c->getDuration());
		c->decreaseDuration(1);
		printf("After %d\n",c->getDuration());
	});
}

inline void update_scheduler(scheduler_t* scheduler){
	scheduler->time++;
	scheduler->total_containers++;
	if(scheduler->total_refused+scheduler->total_accepted!=scheduler->total_containers) {
		std::cerr << "Erro in containers total check\n";
		exit(1);
	}
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


	// if(argc==2) {
	// conn->getNTasks(atoi(argv[1]));
	// Container *c = new Container();
	//const char* task=conn->getNextTask();
	// }
	fflush(stdin);
	while(message_count--) {
		Container *c = new Container();
		c->setTask(conn->getNextTask());
		scheduler->containers.push_back(c);
		std::cout << *c<<"\n";
		// reduce all the containers time by 1, when 0, removes it.
		update_lifetime(scheduler);
		// allocate the new container in the data center.

		// update the objective functions
		getchar();
		update_scheduler(scheduler);
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
