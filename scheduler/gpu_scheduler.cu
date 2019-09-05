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

#include "allocator/links/links_allocator.cuh"

#include "objective_functions/fragmentation.hpp"
#include "objective_functions/footprint.hpp"

void setup(int argc, char** argv, Builder* builder, options_t* options){
	std::string topology = "fat_tree";
	std::string multicriteria_method = "ahpg";
	std::string clustering_method = "mcl";
	std::string standard = "none";
	std::string debug="info";
	std::string data_type="flat";
	int topology_size=10;
	// dont show the help by default. Use `-h or `--help` to enable it.
	bool showHelp = false;
	unsigned int test_type=0;
	unsigned int request_size=0;
	unsigned int bw = 0;
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
	           | clara::detail::Opt( data_type, "Data Type")["--data-type"]("flat | frag | bw")
	           | clara::detail::Opt( bw, "bandwidth")["--bw"]("Only used in test 4");

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

	if(standard=="none" || standard=="ff" || standard=="first_fit" || standard=="bf" || standard=="best_fit"  || standard=="wf" || standard=="worst_fit") {
		options->standard=standard;
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

	if(data_type=="flat") {
		if(builder->getMulticriteria()!=NULL)
			builder->getMulticriteria()->setType(0);
		if(builder->getMulticriteriaClustered()!=NULL)
			builder->getMulticriteriaClustered()->setType(0);
		options->data_type = 0;
	}else if(data_type=="frag" | data_type=="fragmentation") {
		if(builder->getMulticriteria()!=NULL)
			builder->getMulticriteria()->setType(1);
		if(builder->getMulticriteriaClustered()!=NULL)
			builder->getMulticriteriaClustered()->setType(1);
		options->data_type = 1;
	}else if(data_type=="bw" | data_type=="bandwidth") {
		if(builder->getMulticriteria()!=NULL)
			builder->getMulticriteria()->setType(2);
		if(builder->getMulticriteriaClustered()!=NULL)
			builder->getMulticriteriaClustered()->setType(2);
		options->data_type = 2;
	}else{
		SPDLOG_ERROR("Invalid data type");
	}
	if(builder->getMulticriteria()!=NULL)
		builder->getMulticriteria()->readJson();
	if(builder->getMulticriteriaClustered()!=NULL)
		builder->getMulticriteriaClustered()->readJson();

	options->bw = bw;

	options->current_time=0;
	// Load the Topology
	std::string path="datacenter/json/"+topology+"/"+std::to_string(topology_size)+".json";
	builder->parser(path.c_str());
}

inline void calculateObjectiveFunction(objective_function_t *obj, consumed_resource_t consumed, total_resources_t total, int low){
	int high = low+1;
	obj->time = consumed.time;
	obj->dc_fragmentation = ObjectiveFunction::Fragmentation::datacenter( consumed, total);
	obj->link_fragmentation = ObjectiveFunction::Fragmentation::link(consumed, total);
	obj->vcpu_footprint = ObjectiveFunction::Footprint::vcpu(consumed, total, low, high);
	obj->ram_footprint = ObjectiveFunction::Footprint::ram(consumed, total, low, high);
	obj->link_footprint = ObjectiveFunction::Footprint::link(consumed,total);
	obj->footprint = ObjectiveFunction::Footprint::footprint(consumed, total, low, high);
}

inline void logTask(scheduler_t* scheduler,Task* task, std::string multicriteria, total_resources_t* total_resources){
	std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span =  std::chrono::duration_cast<std::chrono::duration<double> >( now - scheduler->start);
	spdlog::get("task_logger")->info("{} {} {} {} {} {} {} {} {} {}", multicriteria, task->getSubmission(), task->getId(), task->getDelay(), task->taskUtility(), task->linkUtility(), time_span.count(), task->getDelayDC(), task->getDelayLink(),task->getBandwidthAllocated()/total_resources->total_bandwidth);
}

inline void logDC(objective_function_t *objective,std::string method, float total_bandwidth){
	spdlog::get("dc_logger")->info("{} {} {} {} {} {} {} {}", method, objective->time,    objective->dc_fragmentation,  objective->vcpu_footprint, objective->ram_footprint, objective->link_fragmentation, objective->link_footprint, (objective->fail_bandwidth/total_bandwidth));
}

inline void delete_tasks(scheduler_t* scheduler, Builder* builder, options_t* options, consumed_resource_t* consumed, objective_function_t* objective){
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
		spdlog::debug("Scheduler Time {}\t\tDeleting task {}", options->current_time, current->getId());
		//builder->getTopology()->listTopology();
		objective->fail_bandwidth-=current->getBandwidthAllocated();
		// if(options->standard=="none") {
		if(options->test_type==4) {
			Allocator::freeAllResources(
				/* The task to be removed*/
				current,
				/* The consumed DC status*/
				consumed,
				builder,
				current->getAllocatedTime(),
				options->current_time
				);
		}else{
			Allocator::freeHostResource(
				current,
				consumed,
				builder,
				current->getAllocatedTime(),
				options->current_time
				);
		}
		delete(current);

		//builder->getTopology()->listTopology();
	}

	current = NULL;
}

inline void allocate_tasks(scheduler_t* scheduler, Builder* builder, options_t* options, consumed_resource_t* consumed, total_resources_t* total_dc, objective_function_t* objective){
	spdlog::debug("allocate task");
	bool allocation_success = false;
	bool allocation_link_success = false;
	Task* current = NULL;
	int total_delay = 0;
	int delay=1;

	std::chrono::duration<double> time_span_links;
	std::chrono::duration<double> time_span_allocator;
	while(true) {
		allocation_success=false;
		allocation_link_success=false;
		if(scheduler->tasks_to_allocate.empty()) {
			spdlog::debug("empty allocate queue");
			break;
		}

		current = scheduler->tasks_to_allocate.top();

		if( current->getSubmission()+current->getDelay() != options->current_time) {
			spdlog::debug("request in advance time submission {} delay {} scheduler time {}",current->getSubmission(), current->getDelay(),options->current_time);
			break;
		}

		scheduler->tasks_to_allocate.pop();

		spdlog::debug("Check if request {} fit in DC",current->getId());
		// allocate the new task in the data center.
		std::chrono::high_resolution_clock::time_point allocator_start = std::chrono::high_resolution_clock::now();
		if(options->standard=="none") {
			if( options->clustering_method=="pure_mcl") {
				spdlog::debug("Pure MCL");
				allocation_success=Allocator::mcl_pure(builder);
			} else if( options->clustering_method == "none") {
				spdlog::debug("Naive");
				allocation_success=Allocator::naive(builder, current, consumed, options->current_time);
				spdlog::debug("Naive[x]");
			} else if ( options->clustering_method == "mcl") {
				spdlog::debug("MCL + MULTICRITERIA");
				allocation_success=Allocator::multicriteria_clusterized(builder, current, consumed, options->current_time);
				spdlog::debug("MCL + MULTICRITERIA [X]");
			} else if ( options->clustering_method == "all") {
				// allocation_success=Allocator::all();
			} else {
				SPDLOG_ERROR("Invalid type of allocation method");
				exit(1);
			}
		}else{
			if(options->standard=="ff" || options->standard=="first_fit") {
				allocation_success=Allocator::firstFit(builder, current, consumed, options->current_time);
			}else if(options->standard=="bf" || options->standard=="best_fit") {
				allocation_success=Allocator::bestFit(builder, current, consumed, options->current_time);
			}else if(options->standard=="wf" || options->standard=="worst_fit") {
				allocation_success=Allocator::worstFit(builder, current, consumed, options->current_time);
			}else{
				SPDLOG_ERROR("Invalid type of standard allocation method");
				exit(1);
			}
		}
		std::chrono::high_resolution_clock::time_point allocator_end = std::chrono::high_resolution_clock::now();

		time_span_allocator =  std::chrono::duration_cast<std::chrono::duration<double> >(allocator_end - allocator_start);

		//if(allocation_success && options->standard=="none") {
		if(allocation_success && options->test_type==4) {
			std::chrono::high_resolution_clock::time_point links_start = std::chrono::high_resolution_clock::now();

			// builder->getTopology()->listTopology();
			// allocation_success=Allocator::links_allocator(builder, current, consumed);
			spdlog::debug("links allocator");
			allocation_link_success=Allocator::links_allocator_cuda(builder, current, consumed, options->current_time, options->current_time + current->getDuration());
			spdlog::debug("links allocator [x]");
			// builder->getTopology()->listTopology();
			std::chrono::high_resolution_clock::time_point links_end = std::chrono::high_resolution_clock::now();

			time_span_links =  std::chrono::duration_cast<std::chrono::duration<double> >(links_end - links_start);
			if(!allocation_link_success) {
				spdlog::info("\tRequest dont fit in links");
			}
		}
		if(!allocation_success) {
			spdlog::info("\trequest dont fit in allocation");
		}

		spdlog::debug("\t\tChecking the success of allocation");
		if(!allocation_success || (!allocation_link_success && options->test_type==4)) {
			spdlog::debug("\t\t\tAllocation not succeeded\n");
			// if(!allocation_success || (!allocation_link_success && options->standard=="none")) {
			if(!scheduler->tasks_to_delete.empty()) {
				Task* first_to_delete = scheduler->tasks_to_delete.top();

				delay = ((first_to_delete->getDuration() + first_to_delete->getAllocatedTime()) - ( current->getSubmission() + current->getDelay() ));
			}

			spdlog::debug("added delay {} in request",delay);
			current->addDelay(delay);

			scheduler->tasks_to_allocate.push(current);

			if(!allocation_success) {
				current->addDelayDC(delay);
			}
			// else if(!allocation_link_success && options->standard=="none"){
			else if(!allocation_link_success) {
				current->addDelayLink(delay);
			}

			total_delay+=current->getDelay();

			spdlog::info("\tTask {} can't be allocated, added delay of {} next try in scheduler time {}", current->getId(), delay, current->getSubmission()+current->getDelay() );
		}else{
			spdlog::debug("\t\t\tAllocation with Success");
			current->setAllocatedTime(options->current_time);
			scheduler->tasks_to_delete.push(current);
			objective->fail_bandwidth+=current->getBandwidthAllocated();
			if(options->standard=="none") {
				spdlog::get("mb_logger")->info("ALLOCATOR {} {}",options->multicriteria_method,time_span_allocator.count());
				spdlog::get("mb_logger")->info("LINKS {} {}",options->multicriteria_method,time_span_links.count());
				logTask(scheduler, current, options->multicriteria_method,total_dc);
			}else{
				spdlog::get("mb_logger")->info("ALLOCATOR {} {}",options->standard,time_span_allocator.count());
				spdlog::get("mb_logger")->info("LINKS {} {}",options->standard,time_span_links.count());
				logTask(scheduler, current, options->standard,total_dc);
			}
		}
		spdlog::debug("ending the while loop");
	}
	spdlog::debug("allocate task[x]");
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
		spdlog::info("Scheduler Time {}", options->current_time);

		consumed_resources.time = options->current_time;
		// Search the containers to delete
		delete_tasks(scheduler, builder, options, &consumed_resources, &objective);
		// Search the containers in the vector to allocate in the DC
		allocate_tasks(scheduler, builder, options, &consumed_resources, &total_resources, &objective);

		//************************************************//
		//       Print All the metrics information        //
		//************************************************//
		calculateObjectiveFunction(&objective,consumed_resources, total_resources, options->current_time);

		if(options->test_type==2 || options->test_type==4) {
			if(options->standard=="none") {
				logDC(&objective, options->multicriteria_method, total_resources.total_bandwidth);
			}else{
				logDC(&objective, options->standard, total_resources.total_bandwidth);
			}
		}
		//************************************************//
		//************************************************//
		//************************************************//

		options->current_time++;
	}
}

int main(int argc, char **argv){
	auto console = spdlog::stdout_color_mt("console");
	console->set_pattern("[%-6l] %v");
	spdlog::set_default_logger(console);

	spdlog::info("Initialized");
	// Options Struct
	options_t options;
	// Scheduler Struct
	scheduler_t scheduler;

	// Create the Builder
	Builder *builder= new Builder();
	// Parse the command line arguments

	spdlog::info("Build the setup");
	setup(argc,argv,builder,&options);

	spdlog::debug("Build the log path");
	std::string log_str;
	if(options.test_type==1) {
		log_str+="test1.log";
	}
	else if(options.test_type==4) {
		log_str+=options.clustering_method;
		log_str+="-";
		log_str+="pod";
		log_str+=std::to_string(options.request_size);
		log_str+="-bw";
		log_str+=std::to_string(options.bw);
		log_str+="-dt";
		log_str+=std::to_string(options.data_type);
		log_str+=".json";
	}else{
		log_str+=options.clustering_method;
		log_str+="-";
		log_str+=std::to_string(options.test_type);
		log_str+="-size_";
		log_str+=std::to_string(options.request_size);
		log_str+="-tsize_";
		log_str+=std::to_string(options.topology_size);
		log_str+=".log";
	}
	spdlog::info("Generating the loggers");
	auto dc_logger = spdlog::basic_logger_mt("dc_logger","logs/test"+std::to_string(options.test_type)+"/dc-"+log_str);
	auto task_logger =spdlog::basic_logger_mt("task_logger", "logs/test"+std::to_string(options.test_type)+"/request-"+log_str);
	auto micro_bench_logger = spdlog::basic_logger_mt("mb_logger", "logs/test"+std::to_string(options.test_type)+"/micro-bench"+log_str);

	spdlog::flush_every(std::chrono::seconds(30));

	dc_logger->set_pattern("%v");
	task_logger->set_pattern("%v");
	micro_bench_logger->set_pattern("%v");

	spdlog::info("Creating the reader");
	Reader* reader = new Reader();

	spdlog::info("Generate the path");
	// parse all json
	std::string path = "requests/";
	if(options.test_type==1) {
		path+="container/data-";
	} else if(options.test_type==2) {
		path+="googleBorg/google-";
	} else if(options.test_type==3) {
		path+="datacenter/data-";
	} else if(options.test_type==4) {
		path+="container_link/pod";
	}
	path+= std::to_string(options.request_size);
	if(options.test_type==4) {
		path+="-bw";
		path+=std::to_string(options.bw);
	}
	path+=".json";
	spdlog::info("Reading the request json {}", path);
	reader->openDocument(path.c_str());
	std::string message;

	Task * current = NULL;
	spdlog::info("Reading the Tasks");
	while((message=reader->getNextTask())!="eof") {
		// Create new container
		spdlog::debug("New task created");
		current = new Task();
		// Set the resources to the container
		spdlog::debug("Set the variables into task");
		current->setTask(message.c_str());

		current->print();
		// Put the container in the vector
		scheduler.tasks_to_allocate.push(current);
	}
	message.clear();
	delete(reader);

	std::chrono::high_resolution_clock::time_point cluster_time_start = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> cluster_time_span;
	std::chrono::high_resolution_clock::time_point cluster_time_end;

	if(options.clustering_method!="none") {
		spdlog::info("Running the cluster method");

		builder->runClustering(builder->getHosts());

		builder->getClusteringResult();
	}

	if(options.test_type==1 && options.clustering_method == "pure_mcl") {
		cluster_time_end = std::chrono::high_resolution_clock::now();

		cluster_time_span =  std::chrono::duration_cast<std::chrono::duration<double> >(cluster_time_end - cluster_time_start);

		spdlog::get("dc_logger")->info("pure_mcl;{};{};{}", options.topology_size, options.request_size, cluster_time_span.count());
		delete(builder);

		return 0;
	}

	scheduler.start = std::chrono::high_resolution_clock::now();

	spdlog::info("Running the gpu scheduler");
	schedule(builder, &scheduler, &options, 0);

	scheduler.end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span =  std::chrono::duration_cast<std::chrono::duration<double> >(scheduler.end - scheduler.start);

	spdlog::info("Finished the scheduler: method {}, topology size {}, seconds {}",options.multicriteria_method, options.topology_size, time_span.count());

	if(options.test_type==1) {
		cluster_time_end = std::chrono::high_resolution_clock::now();

		cluster_time_span =  std::chrono::duration_cast<std::chrono::duration<double> >(cluster_time_end - cluster_time_start);

		spdlog::get("dc_logger")->info("{};{};{};{};{}", options.multicriteria_method,options.topology_size, options.request_size, time_span.count(), cluster_time_span.count());
	}
	// Free the allocated pointers
	delete(builder);
	return 0;
}
