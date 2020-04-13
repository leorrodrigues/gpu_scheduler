#define SPDLOG_TRACE_ON
#define SPDLOG_DEBUG_ON

#include <chrono>
#include <ctime>
#include <ratio>

#include "builder.cuh"
#include "reader.hpp"
#include "thirdparty/clara.hpp"

#include "allocator/rank_clusterized.cuh"
#include "allocator/pure_mcl.hpp"
#include "allocator/naive.hpp"
#include "allocator/utils.hpp"
#include "allocator/free.hpp"

#include "allocator/links/links_allocator.cuh"

#include "allocator/rank_algorithms/multicriteria/topsis/topsis.cuh"
#include "allocator/rank_algorithms/multicriteria/ahp/ahpg.cuh"
#include "allocator/rank_algorithms/multicriteria/ahp/ahp.hpp"
#include "allocator/rank_algorithms/standard/worstFit.hpp"
#include "allocator/rank_algorithms/standard/bestFit.hpp"

#include "clustering/mclInterface.cuh"

#include "objective_functions/fragmentation.hpp"
#include "objective_functions/footprint.hpp"

void setup(int argc, char** argv, Builder* builder, options_t* options, scheduler_t *scheduler){
	std::string topology = "fat_tree";
	std::string rank_method = "ahpg";
	std::string rank_clustering_method = "ahpg";
	std::string clustering_method = "mcl";
	std::string debug="info";
	std::string data_type="flat";
	std::string cmp = "fcfs";
	std::string test_file_name = "";
	std::string scheduling_type = "online";
	int topology_size=10;
	// dont show the help by default. Use `-h or `--help` to enable it.
	bool showHelp = false;
	bool automatic_start_time = false;
	unsigned int test_type=0;
	unsigned int request_size=0;
	unsigned int bw = 0;
	auto cli = clara::detail::Help(showHelp)
	           | clara::detail::Opt( topology, "topology" )["-t"]["--topology"]("What is the topology type? [ (default) fat_tree | dcell | bcube ]")
	           | clara::detail::Opt( topology_size, "topology size") ["-s"] ["--topology_size"] ("What is the size of the topology? ( default 10 )")
	           | clara::detail::Opt( rank_method, "rank method") ["-r"]["--rank"] ("What is the rank method? [ ahp | (default) ahpg | topsis | best-fit (bf) | worst-fit (wf)")
	           | clara::detail::Opt( clustering_method,"clustering method") ["-c"]["--clustering"] ("What is the clustering method? [ (default) mcl | pure_mcl | none ]")
	           // 0 For no Test
	           // 1 For Pod Test
	           // 2 For Consolidation Test
	           | clara::detail::Opt( test_type, "Type Test")["--test"]("Which type of test you want?")
	           | clara::detail::Opt( request_size, "Request Size")["--request-size"]("Which is the request size?")
	           | clara::detail::Opt( debug, "Debug option")["--debug"]("info | warning | error | debug")
	           | clara::detail::Opt( data_type, "Data Type")["--data-type"]("flat | frag | bw")
	           | clara::detail::Opt( bw, "bandwidth")["--bw"]("Only used in test 4")
	           | clara::detail::Opt( cmp, "comparator")["--cmp"]("Comparator used in tasks [(default) fcfs | spf | sqfmin | sqfmax | safmin | safmax | sdafmin]")
	           | clara::detail::Opt( test_file_name, "test_file_name")["--file_name"]("File name to run the test5 [grenoble | lyon | nancy | nantes]")
	           | clara::detail::Opt( automatic_start_time, "automatic_time")["-a"]["--automatic_time"]("Calculate the start time automaticaly or start from time 0 [true | (default) false]")
	           | clara::detail::Opt(scheduling_type, "scheduling_type")["--scheduling_type"]("Select the scheduling type will be used [(default) online | offline ]");

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

	if(test_file_name != "") {
		options->topology_size = 0; // if the test_file was set, do not use the topology size explicity
	} else{
		if((topology_size<2 || topology_size>48) && topology_size!=0) {
			SPDLOG_ERROR("Invalid topology size ( must be between 4 and 48 )");
			exit(0);
		}else{
			options->topology_size = topology_size;
		}
	}

	Rank *rank = NULL;
	if( rank_method == "ahpg") {
		rank = new AHPG();
	}else if(rank_method == "ahp" ) {
		rank = new AHP();
	}else if(rank_method=="topsis") {
		rank = new TOPSIS();
	} else if(rank_method=="bf" || rank_method=="best-fit") {
		rank = new BestFit();
	} else if(rank_method=="wf" || rank_method=="worst-fit") {
		rank = new WorstFit();
	} else{
		SPDLOG_ERROR("Invalid rank method");
		exit(0);
	}
	builder->setRank(rank);
	options->rank_method = rank_method;

	bool cluster = false;
	Clustering *clustering_ptr = NULL;
	if( clustering_method == "mcl" || clustering_method == "pure_mcl") {
		clustering_ptr = new MCLInterface();
		cluster= true;
	}else if( clustering_method == "none") {
		cluster = false;
	}else{
		SPDLOG_ERROR("Invalid clustering method");
		exit(0);
	}
	builder->setClustering(clustering_ptr);
	options->clustering_method=clustering_method;

	if(cluster) {
		std::cout<<"CLUSTERED METHOD "<<rank_method<<"\n";
		builder->setClusteredRank(rank);
	}

	if(test_type >0 && test_type<=5) {
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

	if(debug=="info") {
		spdlog::set_level(spdlog::level::info);
	} else if(debug=="warning") {
		spdlog::set_level(spdlog::level::warn);
	} else if(debug=="error") {
		spdlog::set_level(spdlog::level::err);
	} else if(debug=="debug") {
		spdlog::set_level(spdlog::level::debug);
	} else if(debug=="critical") {
		spdlog::set_level(spdlog::level::critical);
	} else {
		SPDLOG_ERROR("Invalid Type of Debug Level");
		exit(0);
	}

	if(data_type=="flat") {
		if(builder->getRank()!=NULL)
			builder->getRank()->setType(0);
		if(builder->getRankClustered()!=NULL)
			builder->getRankClustered()->setType(0);
		options->data_type = 0;
	}else if(data_type=="frag" | data_type=="fragmentation") {
		if(builder->getRank()!=NULL)
			builder->getRank()->setType(1);
		if(builder->getRankClustered()!=NULL)
			builder->getRankClustered()->setType(1);
		options->data_type = 1;
	}else if(data_type=="bw" | data_type=="bandwidth") {
		if(builder->getRank()!=NULL)
			builder->getRank()->setType(2);
		if(builder->getRankClustered()!=NULL)
			builder->getRankClustered()->setType(2);
		options->data_type = 2;
	}else{
		SPDLOG_ERROR("Invalid data type");
	}
	if(builder->getRank()!=NULL)
		builder->getRank()->readJson();
	if(builder->getRankClustered()!=NULL)
		builder->getRankClustered()->readJson();

	if(cmp == "fcfs") {
		scheduler->tasks_to_allocate = new FCFS();
	}else if(cmp == "spf") {
		scheduler->tasks_to_allocate = new SPF();
	}else if(cmp == "sqfmin") {
		scheduler->tasks_to_allocate = new SQFMIN();
	}else if(cmp == "sqfmax") {
		scheduler->tasks_to_allocate = new SQFMAX();
	}else if(cmp == "safmin") {
		scheduler->tasks_to_allocate = new SAFMIN();
	}else if("safmax") {
		scheduler->tasks_to_allocate = new SAFMAX();
	}else if("sdafmin") {
		scheduler->tasks_to_allocate = new SDAFMIN();
	}
	options->queue_type = cmp;

	options->bw = bw;

	scheduler->current_time=0;

	options->test_file_name = test_file_name;

	options->automatic_start_time = automatic_start_time;

	options->scheduling_type = scheduling_type;

	std::string dt;
	{
		std::istringstream f(test_file_name);
		getline(f, dt, '-');
	}
	// Load the Topology
	std::string path;
	if(test_file_name == "") {
		path="datacenter/"+topology+"/" + std::to_string(topology_size) + ".json";
	} else {
		path="datacenter/generic/"+dt+".json";
	}
	builder->parser(path.c_str());
}

inline void calculateObjectiveFunction(Builder *builder, objective_function_t *obj, consumed_resource_t consumed, total_resources_t total, int low){
	int high = low+1;
	obj->time = consumed.time;
	obj->dc_fragmentation = ObjectiveFunction::Fragmentation::datacenter(builder, total, low, high);
	obj->link_fragmentation = ObjectiveFunction::Fragmentation::link(consumed, total);
	obj->vcpu_footprint = ObjectiveFunction::Footprint::vcpu(builder, total, low, high);
	obj->ram_footprint = ObjectiveFunction::Footprint::ram(builder,  total, low, high);
	obj->link_footprint = ObjectiveFunction::Footprint::link(consumed,total);
	obj->footprint = ObjectiveFunction::Footprint::footprint(builder, total, low, high);
}

inline void logTask(scheduler_t* scheduler,Task* task, std::string rank, total_resources_t* total_resources){
	std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();

	// calculate the spent time using the start scheduler time as epoch time.
	std::chrono::duration<double> requested_time = std::chrono::duration_cast<std::chrono::duration<double> >(task->getRequestedTime() - scheduler->start);
	std::chrono::duration<double> start_time = std::chrono::duration_cast<std::chrono::duration<double> >(task->getStartTime() - scheduler->start);
	std::chrono::duration<double> stop_time = std::chrono::duration_cast<std::chrono::duration<double> >(task->getStopTime() - scheduler->start);
	//represents the time needed to run the task since the scheduler initiates
	std::chrono::duration<double> time_span =  std::chrono::duration_cast<std::chrono::duration<double> >( now - scheduler->start);

	spdlog::get("task_logger")->critical("{} {} {} {} {} {} {} {} {} {} {}", rank, task->getSubmission(), task->getId(), task->getDelay(), task->taskUtility(), task->linkUtility(), task->getBandwidthAllocated()/total_resources->total_bandwidth, requested_time.count(), start_time.count(), stop_time.count(), time_span.count());
}

inline void logDC(scheduler_t* scheduler, objective_function_t *objective,std::string method, float total_bandwidth, total_resources_t *total){
	std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span =  std::chrono::duration_cast<std::chrono::duration<double> >( now - scheduler->start);

	spdlog::get("dc_logger")->critical("{} {} {} {} {} {} {} {} {} {} {}", method, objective->time,    objective->dc_fragmentation,  objective->vcpu_footprint, objective->ram_footprint, objective->link_fragmentation, objective->link_footprint, (objective->fail_bandwidth/total_bandwidth), total->rejected_tasks, total->accepted_tasks, time_span.count());
}

inline void delete_tasks(scheduler_t* scheduler, Builder* builder, options_t* options, consumed_resource_t* consumed, objective_function_t* objective, total_resources_t* total_dc){
	Task* current = NULL;
	while(true) {
		if(scheduler->tasks_to_delete.empty()) {
			break;
		}
		current=scheduler->tasks_to_delete.top();
		if( (current->getDuration()-current->getEarlyDuration()) + current->getAllocatedTime() > scheduler->current_time) {
			break;
		}
		scheduler->tasks_to_delete.pop();
		//At this moment, the task has already been stopped, so update the corresponding time
		current->setStopTime();
		//Iterate through the PODs of the TASK, and erase each of one.
		spdlog::debug("Scheduler Time {}. Deleting task {}", scheduler->current_time, current->getId());
		objective->fail_bandwidth-=current->getBandwidthAllocated();
		if(options->bw > 0) { //represents that the tests use bw
			Allocator::freeAllResources(
				/* The task to be removed*/
				current,
				/* The consumed DC status*/
				consumed,
				builder,
				current->getAllocatedTime(),
				current->getAllocatedTime() + current->getDuration()
				);
		}else{
			Allocator::freeHostResource(
				current,
				builder,
				current->getAllocatedTime(),
				current->getAllocatedTime() + current->getDuration()
				);
		}
		//After remove the task from the DC log their informations
		logTask(scheduler, current, options->rank_method,total_dc);
		//Delete the task completly
		delete(current);
	}
	current = NULL;
}

inline void allocate_tasks(scheduler_t* scheduler, Builder* builder, options_t* options, consumed_resource_t* consumed, total_resources_t* total_dc, objective_function_t* objective){
	spdlog::debug("allocate task");
	bool allocation_success = false;
	bool allocation_link_success = false;
	Task* current = NULL;
	int delay = 0;

	while(true) {
		allocation_success=false;
		allocation_link_success=false;
		if(scheduler->tasks_to_allocate->empty()) {
			spdlog::debug("empty allocate queue");
			break;
		}
		current = scheduler->tasks_to_allocate->top();
		if(current->getSubmission() == scheduler->current_time) {
			//this means that is the first time that this task is checked, set the requested_time.
			current->setRequestedTime();
		}

		if( current->getSubmission()+current->getDelay() > scheduler->current_time) {
			spdlog::debug("request in advance time submission {} delay {} scheduler time {}",current->getSubmission(), current->getDelay(),scheduler->current_time);
			break;
		}
		scheduler->tasks_to_allocate->pop();
		++total_dc->total_tasks;
		spdlog::info("Check if request {} fit in DC, submission {} delay {} duration {} early {} deadline {}",current->getId(), current->getSubmission(), current->getDelay(), current->getDuration(), current->getEarlyDuration(), current->getDeadline());
		// allocate the new task in the data center.
		if( options->clustering_method=="pure_mcl") {
			spdlog::debug("Pure MCL");
			allocation_success=Allocator::mcl_pure(builder);
		} else if( options->clustering_method == "none") {
			spdlog::debug("Naive [ ]");
			allocation_success=Allocator::naive(builder, current, consumed, scheduler->current_time, options->scheduling_type);
			spdlog::debug("Naive [x]");
		} else if ( options->clustering_method == "mcl") {
			spdlog::debug("MCL + Rank [ ]");
			allocation_success=Allocator::rank_clusterized(builder, current, consumed, scheduler->current_time, options->scheduling_type);
			spdlog::debug("MCL + Rank [x]");
		} else {
			SPDLOG_ERROR("Invalid type of allocation method");
			exit(1);
		}

		if(allocation_success && options->bw>0) {
			allocation_link_success=Allocator::links_allocator_cuda(builder, current, consumed, scheduler->current_time, scheduler->current_time + current->getDuration());
		}
		if(!allocation_success || (!allocation_link_success && options->bw>0)) {
			// If the request is not suitable in the DC, check the scheduling type to make the right decision about the task
			if(options->scheduling_type == "online") {
				++total_dc->rejected_tasks;
			} else if(options->scheduling_type == "offline") {
				// At first, we need to calculate the new delay to apply to the request
				delay=1;
				if(!scheduler->tasks_to_delete.empty()) {
					Task* first_to_delete = scheduler->tasks_to_delete.top();
					delay = ((first_to_delete->getDuration() + first_to_delete->getAllocatedTime()) - ( current->getSubmission() + current->getDelay() ));
				}
				if(delay<=0) delay=1; //to be sure that the delay will be at least plus 1
				current->addDelay(delay);
				// With the calculated delay, check if the new applyed time to the task + the execution time is higher than the task deadline sum the rejected_tasks, reinsert it on the queue otherwise.
				if(current->getDeadline() != 0 && (current->getSubmission() + current->getDelay() + current->getDuration() - current->getEarlyDuration()) > current->getDeadline()) {
					++total_dc->rejected_tasks; //to reject the task
				} else {
					scheduler->tasks_to_allocate->push(current); // try to allocate the task again in the future
				}
			} else {
				SPDLOG_ERROR("Wrong scheduling type selected!\nExiting...");
				exit(0);
			}
		} else {
			// The task was successfully allocated, so update the metrics
			current->setStartTime();
			++total_dc->accepted_tasks;
			scheduler->tasks_to_delete.push(current);
			objective->fail_bandwidth += current->getBandwidthAllocated();
		}
	}
}

void schedule(Builder* builder,  scheduler_t* scheduler, options_t* options, int message_count){
	const int total_tasks = scheduler->tasks_to_allocate->size();
	//Create the variable to store all the data center resource
	total_resources_t total_resources;

	spdlog::debug("Set data center resources");
	builder->setDataCenterResources(&total_resources);

	consumed_resource_t consumed_resources;
	objective_function_t objective;

	spdlog::debug("Start scheduler main loop");
	while(
		!scheduler->tasks_to_allocate->empty() ||
		!scheduler->tasks_to_delete.empty()
		) {
		spdlog::info("Scheduler Time {}", scheduler->current_time);
		spdlog::info("Tasks to allocate {} and to delete {}", scheduler->tasks_to_allocate->size(), scheduler->tasks_to_delete.size());

		consumed_resources.time = scheduler->current_time;
		// Search the containers to delete
		delete_tasks(scheduler, builder, options, &consumed_resources, &objective, &total_resources);
		// Search the containers in the vector to allocate in the DC
		allocate_tasks(scheduler, builder, options, &consumed_resources, &total_resources, &objective);

		//************************************************//
		//       Print All the metrics information        //
		//************************************************//
		//************************************************//
		spdlog::info("Calculating the objective functions");
		//First update the active servers
		calculateObjectiveFunction(builder, &objective,consumed_resources, total_resources, scheduler->current_time);
		spdlog::debug("Objective functions calculated with success");

		if(options->test_type == 2 || options->test_type == 4 || options->test_type == 5) {
			logDC(scheduler, &objective, options->rank_method, total_resources.total_bandwidth, &total_resources);
		}
		//************************************************//
		//************************************************//
		//************************************************//

		scheduler->current_time++;
		spdlog::debug("Scheduler time {} end\n",scheduler->current_time);
	}
	spdlog::info("Tasks rejected {} of {}, Reject percentage {}%",total_resources.rejected_tasks, total_resources.total_tasks, (total_resources.rejected_tasks/(total_resources.total_tasks*1.0))*100);
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
	setup(argc,argv,builder,&options, &scheduler);

	spdlog::debug("Build the log path");
	std::string log_str;

	log_str+=options.clustering_method;
	if(options.test_type == 4) {
		log_str+="-pod";
		log_str+=std::to_string(options.request_size);
		log_str+="-bw";
		log_str+=std::to_string(options.bw);
		log_str+="-dt";
		log_str+=std::to_string(options.data_type);
		log_str+=".log";
	} else if(options.test_type == 5) {
		log_str+= "-"+options.test_file_name;
		log_str+= "-"+options.queue_type;
	} else {
		log_str+="-size_";
		log_str+=std::to_string(options.request_size);
		log_str+="-tsize_";
		log_str+=std::to_string(options.topology_size);
		log_str+=".log";
	}
	spdlog::info("Generating the loggers");
	auto dc_logger = spdlog::basic_logger_mt("dc_logger","logs/test"+std::to_string(options.test_type)+"/dc-"+log_str+"-"+options.scheduling_type+".log");
	auto task_logger =spdlog::basic_logger_mt("task_logger", "logs/test"+std::to_string(options.test_type)+"/request-"+log_str+"-"+options.scheduling_type+".log");

	spdlog::flush_every(std::chrono::seconds(180));

	dc_logger->set_pattern("%v");
	task_logger->set_pattern("%v");

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
	} else if(options.test_type==5) {
		path+="temporal/";
	}
	path+= std::to_string(options.request_size);
	if(options.test_type == 4) {
		path+="-bw";
		path+=std::to_string(options.bw);
	} else if(options.test_type == 5) {
		path.pop_back();
		path+=options.test_file_name;
	}
	path+=".json";
	spdlog::info("Reading the request json {}", path);
	reader->openDocument(path.c_str());
	std::string message;

	Task * current = NULL;
	spdlog::info("Reading the Tasks");
	int total_tasks = reader->getTasksSize(), current_task_index = 0;
	while((message=reader->getNextTask())!="eof") {
		// Create new container
		spdlog::debug("New task created");
		current = new Task();
		// Set the resources to the container
		spdlog::debug("Set the variables into task");
		current->setTask(message.c_str());

		current->print();
		// Put the container in the vector
		scheduler.tasks_to_allocate->push(current);
		spdlog::info("Read {} task of {} tasks", ++current_task_index, total_tasks);
	}
	message.clear();
	delete(reader);

	// After get the tasks, move the start time of the scheduler to the first request to allocate minus 1
	if(options.automatic_start_time) {
		scheduler.current_time = scheduler.tasks_to_allocate->top()->getSubmission()-1;
	}


	std::chrono::high_resolution_clock::time_point cluster_time_start = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> cluster_time_span;
	std::chrono::high_resolution_clock::time_point cluster_time_end;

	if(options.clustering_method!="none") {
		builder->runClustering();
		builder->getClusteringResult();
		if(builder->getClusteringResultSize() == 0) {
			SPDLOG_ERROR("There aren't any groups formed! --> Error in {} algorithm",options.clustering_method);
			exit(0);
		}
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

	spdlog::info("Finished the scheduler: method {}, topology size {}, seconds {}",options.rank_method, options.topology_size, time_span.count());

	if(options.test_type==1) {
		cluster_time_end = std::chrono::high_resolution_clock::now();

		cluster_time_span =  std::chrono::duration_cast<std::chrono::duration<double> >(cluster_time_end - cluster_time_start);

		spdlog::get("dc_logger")->info("{};{};{};{};{}", options.rank_method,options.topology_size, options.request_size, time_span.count(), cluster_time_span.count());
	}
	// Free the allocated pointers
	delete(builder);
	return 0;
}
