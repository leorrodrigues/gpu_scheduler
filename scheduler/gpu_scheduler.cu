#define SPDLOG_TRACE_ON
#define SPDLOG_DEBUG_ON

#include <chrono>
#include <ctime>
#include <ratio>

#include "builder.cuh"
#include "reader.hpp"
#include "thirdparty/clara.hpp"

#include "allocator/standard/bestFit.hpp"
#include "allocator/standard/firstFit.hpp"
#include "allocator/standard/worstFit.hpp"

#include "allocator/multicriteria_clusterized.cuh"
#include "allocator/pure_mcl.hpp"
#include "allocator/naive.hpp"
#include "allocator/utils.hpp"
#include "allocator/free.hpp"
#include "allocator/all.cuh"

#include "allocator/links/links_allocator.hpp"

#include "objective_functions/fragmentation.hpp"
#include "objective_functions/footprint.hpp"

void setup(int argc, char** argv, Builder* builder, options_t* options){
	std::string topology = "fat_tree";
	std::string multicriteria_method = "ahpg";
	std::string clustering_method = "mcl";
	std::string standard = "none";
	std::string debug="info";
	std::string data_type="frag";
	int topology_size=10;
	// dont show the help by default. Use `-h or `--help` to enable it.
	bool showHelp = false;
	unsigned int test_type=0;
	unsigned int request_size=0;
	auto cli = clara::detail::Help(showHelp)
	           | clara::detail::Opt( topology, "topology" )["-t"]["--topology"]("What is the topology type? [ (default) fat_tree | dcell | bcube ]")
	           | clara::detail::Opt( topology_size, "topology size") ["-s"] ["--topology_size"] ("What is the size of the topology? ( default 10 )")
	           | clara::detail::Opt( multicriteria_method, "multicriteria method") ["-m"]["--multicriteria"] ("What is the multicriteria method? [ ahp | (default) ahpg | topsis]")
	           | clara::detail::Opt( clustering_method,"clustering method") ["-c"]["--clustering"] ("What is the clustering method? [ (default) mcl | pure_mcl | none ]")
	           // 0 For no Test
	           // 1 For Pod Test
	           // 2 For Consolidation Test
	           | clara::detail::Opt( test_type, "Type Test")["--test"]("Which type of test you want?")
	           | clara::detail::Opt( request_size, "Request Size")["--request-size"]("Which is the request size?")
	           | clara::detail::Opt( standard, "Standard Allocation")["--standard-allocation"]("What is the standard allocation method? [best_fit (bf) | worst_fit (wf) | first_fit (ff) ]")
	           | clara::detail::Opt( debug, "Debug option")["--debug"]("info | warning | error | debug")
	           | clara::detail::Opt( data_type, "Data Type")["--data-type"]("flat | frag");

	auto result = cli.parse( clara::detail::Args( argc, argv ) );

	if( !result ) {
		SPDLOG_ERROR("Error in command line: {}", result.errorMessage());
		exit(1);
	}

	if ( showHelp ) {
		std::cout << cli << std::endl;
		exit(0);
	}

	if( topology != "fat_tree" && topology != "dcell" && topology != "bcube" ) {
		SPDLOG_ERROR("Invalid entered topology");
		exit(0);
	}else{
		options->topology_type=topology;
	}

	if( (topology_size<2 || topology_size>48) && topology_size!=0) {
		SPDLOG_ERROR("Invalid topology size ( must be between 4 and 48 )");
		exit(0);
	}else{
		options->topology_size=topology_size;
	}

	if( multicriteria_method == "ahpg") {
		builder->setAHPG();
	}else if(multicriteria_method == "ahp" ) {
		builder->setAHP();
	}else if(multicriteria_method=="topsis") {
		builder->setTOPSIS();
	} else{
		SPDLOG_ERROR("Invalid multicriteria method");
		exit(0);
	}
	options->multicriteria_method=multicriteria_method;

	bool cluster = false;
	if( clustering_method == "mcl" ) {
		builder->setMCL();
		cluster= true;
	}else if( clustering_method == "pure_mcl") {
		builder->setMCL();
		cluster = true;
	}else if( clustering_method == "none") {
		cluster = false;
	}else{
		SPDLOG_ERROR("Invalid clustering method");
		exit(0);
	}
	options->clustering_method=clustering_method;

	if(cluster) {
		if(multicriteria_method=="ahp") {
			builder->setClusteredAHP();
		}else if(multicriteria_method=="ahpg") {
			builder->setClusteredAHPG();
		}else if(multicriteria_method=="topsis") {
			builder->setClusteredTOPSIS();
		}else{
			SPDLOG_ERROR("Invalid multicriteria method");
			exit(0);
		}
	}

	if(test_type >0 && test_type<=4) {
		options->test_type=test_type;
	}else{
		SPDLOG_ERROR("Invalid Type of test: {}", test_type);
		exit(0);
	}

	if(request_size<=0 && request_size>=22) {
		SPDLOG_ERROR("Invalid Size of Request");
		exit(0);
	}else{
		options->request_size=request_size;
	}

	if(standard=="ff" || standard=="first_fit") {
		options->standard=1;
	}else if(standard=="bf" || standard=="best_fit") {
		options->standard=2;
	}else if(standard=="wf" || standard=="worst_fit") {
		options->standard=3;
	}else if(standard=="none") {
	}else{
		SPDLOG_ERROR("Invalid Type of standard allocation");
		exit(0);
	}

	if(debug=="info") {
		spdlog::set_level(spdlog::level::info);
	}else if(debug=="warning") {
		spdlog::set_level(spdlog::level::warn);
	}else if(debug=="error") {
		spdlog::set_level(spdlog::level::err);
	}else if(debug=="debug") {
		spdlog::set_level(spdlog::level::debug);

	}else{
		SPDLOG_ERROR("Invalid Type of Debug Level");
		exit(0);
	}

	if(data_type=="frag") {
		if(builder->getMulticriteria()!=NULL)
			builder->getMulticriteria()->setType(1);
		if(builder->getMulticriteriaClustered()!=NULL)
			builder->getMulticriteriaClustered()->setType(1);
	}else if(data_type=="flat") {
		if(builder->getMulticriteria()!=NULL)
			builder->getMulticriteria()->setType(0);
		if(builder->getMulticriteriaClustered()!=NULL)
			builder->getMulticriteriaClustered()->setType(0);
	}else{
		SPDLOG_ERROR("Invalid data type");
	}

	options->current_time=0;
	// Load the Topology
	std::string path="datacenter/json/"+topology+"/"+std::to_string(topology_size)+".json";
	builder->parser(path.c_str());
}

inline objective_function_t calculateObjectiveFunction(consumed_resource_t consumed, total_resources_t total){
	// printf("Started Calculate Objective Function\n");
	objective_function_t obj;
	// printf("Updating time\n");
	obj.time = consumed.time;

	// printf("Fragmentation\n");
	obj.dc_fragmentation = ObjectiveFunction::Fragmentation::datacenter(consumed, total);

	obj.link_fragmentation = ObjectiveFunction::Fragmentation::link(consumed,total);

	// printf("vcpu footprint\n");
	obj.vcpu_footprint = ObjectiveFunction::Footprint::vcpu(consumed, total);

	// printf("ram footprint\n");
	obj.ram_footprint = ObjectiveFunction::Footprint::ram(consumed, total);

	obj.link_footprint = ObjectiveFunction::Footprint::link(consumed,total);

	// printf("footprint\n");
	obj.footprint = ObjectiveFunction::Footprint::footprint(consumed, total);
	return obj;
}

inline void logTask(scheduler_t *scheduler, Task* task, unsigned int time, std::string multicriteria){
	spdlog::get("task_logger")->info("{} {} {} {} {} {} {}", multicriteria, task->getSubmission(), task->getId(), task->getDelay(), task->taskUtility(), task->linkUtility(), time);
}

inline void logDC(scheduler_t *scheduler, objective_function_t *objective,std::string multicriteria){
	spdlog::get("dc_logger")->info("{} {} {} {} {} {} {}", multicriteria, objective->time,    objective->dc_fragmentation,  objective->vcpu_footprint, objective->ram_footprint, objective->link_fragmentation, objective->link_footprint);
}

inline void delete_tasks(scheduler_t* scheduler, Builder* builder, options_t* options, consumed_resource_t* consumed){
	Task* current = NULL;

	while(true) {

		if(scheduler->tasks_to_delete.empty()) {
			break;
		}

		current=scheduler->tasks_to_delete.top();

		if( current->getDuration() + current->getAllocatedTime() != options->current_time) {
			break;
		}

		scheduler->tasks_to_delete.pop();

		// if(  [ current->getId() ]!=NULL ) {
		//Iterate through the PODs of the TASK, and erase each of one.
		// printf("Scheduler Time %d\n\tDeleting task %d\n", options->current_time, current->getId());

		Allocator::freeAllResources(
			/* The task to be removed*/
			current,
			/* The consumed DC status*/
			consumed,
			builder
			);

		delete(current);
	}

	// builder->getTopology()->listTopology();

	current = NULL;
}

inline void allocate_tasks(scheduler_t* scheduler, Builder* builder, options_t* options, consumed_resource_t* consumed, total_resources_t* total_dc){

	bool allocation_success = false;
	Task* current = NULL;
	int total_delay = 0;
	int delay=1;

	while(true) {
		if(scheduler->tasks_to_allocate.empty()) {
			break;
		}

		current = scheduler->tasks_to_allocate.top();

		if( current->getSubmission()+current->getDelay() != options->current_time) {
			break;
		}

		scheduler->tasks_to_allocate.pop();

		if(Allocator::checkFit(total_dc, consumed,current)!=0) {
			// allocate the new task in the data center.
			if(options->standard==0) {
				if( options->clustering_method=="pure_mcl") {
					allocation_success=Allocator::mcl_pure(builder);
				} else if( options->clustering_method == "none") {
					allocation_success=Allocator::naive(builder,current, consumed);
				} else if ( options->clustering_method == "mcl") {
					allocation_success=Allocator::multicriteria_clusterized(builder, current, consumed);
				} else if ( options->clustering_method == "all") {
					// allocation_success=Allocator::all();
				} else {
					std::cerr << "(gpu_scheduler) Invalid type of allocation method\n";
					exit(1);
				}
			}else{
				if(options->standard==1) {
					allocation_success=Allocator::firstFit(builder, current, consumed);
				}else if(options->standard==2) {
					allocation_success=Allocator::bestFit(builder, current, consumed);
				}else if(options->standard==3) {
					allocation_success=Allocator::worstFit(builder, current, consumed);
				}else{
					std::cerr << "(gpu_scheduler) Invalid type of standard allocation method\n";
					exit(1);
				}
			}
			if(allocation_success) {
				// builder->getTopology()->listTopology();
				allocation_success=Allocator::links_allocator(builder, current, consumed);
				// builder->getTopology()->listTopology();
			}
		}else{
			allocation_success=false;
		}

		if(!allocation_success) {

			if(!scheduler->tasks_to_delete.empty()) {
				Task* first_to_delete = scheduler->tasks_to_delete.top();

				delay = (first_to_delete->getDuration() + first_to_delete->getAllocatedTime()) - ( current->getSubmission() + current->getDelay() );
			}

			current->addDelay(delay);

			scheduler->tasks_to_allocate.push(current);

			total_delay+=current->getDelay();

			// printf("\tTask %d Added Delay in time %d\n", current->getId(), current->getSubmission()+current->getDelay() );
		}else{
			// printf("\tTask %d Allocated in time %d\n", current->getId(), current->getSubmission()+current->getDelay() );
			current->setAllocatedTime(options->current_time);
			scheduler->tasks_to_delete.push(current);
		}
		logTask(scheduler, current, options->current_time, options->multicriteria_method);
	}
	// printf("total_delay,%d,%d\n", options->current_time,total_delay);
}

void schedule(Builder* builder,  scheduler_t* scheduler, options_t* options, int message_count){
	const int total_tasks = scheduler->tasks_to_allocate.size();
	//Create the variable to store all the data center resource
	total_resources_t total_resources;

	builder->setDataCenterResources(&total_resources);

	consumed_resource_t consumed_resources;
	objective_function_t objective;

	while(
		!scheduler->tasks_to_allocate.empty() ||
		!scheduler->tasks_to_delete.empty()
		) {

		consumed_resources.time = options->current_time;
		// if(options->current_time==options->end_time) break;
		// std::cout<<"Scheduler Time "<< options->current_time<<"\n";
		// std::cout<<"message_count "<<message_count<<"\n";

		// Search the containers to delete
		delete_tasks(scheduler, builder, options, &consumed_resources);
		// Search the containers in the vector to allocate in the DC
		allocate_tasks(scheduler, builder, options, &consumed_resources, &total_resources);

		//************************************************//
		//       Print All the metrics information        //
		//************************************************//
		objective=calculateObjectiveFunction(consumed_resources, total_resources);

		if(options->test_type==2 || options->test_type==4) {
			logDC(scheduler, &objective, options->multicriteria_method);
		}
		//************************************************//
		//************************************************//
		//************************************************//

		options->current_time++;
	}
}

int main(int argc, char **argv){
	// Options Struct
	options_t options;
	// Scheduler Struct
	scheduler_t scheduler;

	// Create the Builder
	Builder *builder= new Builder();
	// Parse the command line arguments

	setup(argc,argv,builder,&options);


	std::string log_str(std::to_string(options.test_type));
	log_str+="-size_";
	log_str+=std::to_string(options.request_size);
	log_str+="-tsize_";
	log_str+=std::to_string(options.topology_size);
	log_str+=".txt";

	auto dc_logger = spdlog::basic_logger_mt("dc_logger","logs/dc-test_"+log_str);
	auto task_logger =spdlog::basic_logger_mt("task_logger", "logs/task-test_"+log_str);

	dc_logger->set_pattern("%v");
	task_logger->set_pattern("%v");

	auto console = spdlog::stdout_color_mt("console");
	console->set_pattern("[%-6l] %v");
	spdlog::set_default_logger(console);


	Reader* reader = new Reader();
	// parse all json
	std::string path = "requests/";
	if(options.test_type==1) {
		path+="container/data-";
	} else if(options.test_type==2) {
		path+="googleBorg/google-";
	} else if(options.test_type==3) {
		path+="datacenter/data-";
	} else if(options.test_type==4) {
		path+="container_link/requests-";
	}
	path+= std::to_string(options.request_size);
	path+=".json";
	reader->openDocument(path.c_str());
	std::string message;

	Task * current = NULL;
	spdlog::info("Reading the Tasks");
	while((message=reader->getNextTask())!="eof") {
		// Create new container
		current = new Task();
		// Set the resources to the container
		current->setTask(message.c_str());

		current->print();
		// Put the container in the vector
		scheduler.tasks_to_allocate.push(current);
		// std::cout<<*current<<"\n";
	}
	message.clear();
	delete(reader);

	if(options.clustering_method!="none") {
		spdlog::info("Running the cluster method");
		builder->runClustering(builder->getHosts());

		builder->getClusteringResult();
	}
	// Scalability Test or Objective Function Test
	// force cout to not print in cientific notation
	std::cout<<std::fixed;

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	spdlog::info("Running the gpu scheduler");
	schedule(builder, &scheduler, &options, 0);

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);

	spdlog::info("Finished the scheduler: method {}, topology size {}, seconds {}",options.multicriteria_method, options.topology_size, time_span.count());

	// Free the allocated pointers

	delete(builder);
	return 0;
}
