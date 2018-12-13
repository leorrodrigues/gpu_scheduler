
#include <iostream>
#include <chrono>
#include <ctime>
#include <ratio>

#include "types.hpp"
#include "reader.hpp"
#include "builder.cuh"
#include "thirdparty/clara.hpp"

#include "allocator/ahp_clusterized.cuh"
#include "allocator/pure_mcl.hpp"
#include "allocator/naive.hpp"
#include "allocator/free.hpp"
#include "allocator/all.cuh"
#include "allocator/dc.cuh"

#include "objective_functions/fragmentation.hpp"
#include "objective_functions/footprint.hpp"

void setup(int argc, char** argv, Builder* builder, scheduler_t *scheduler, options_t* options){
	std::string topology = "fat_tree";
	std::string multicriteria_method = "ahpg_clusterized";
	std::string clustering_method = "mcl";
	std::string allocation_type = "naive";
	int topology_size=10;
	int start_scheduler_time=0;
	int end_time=-1;
	// dont show the help by default. Use `-h or `--help` to enable it.
	bool showHelp = false;
	int test_type=0;
	int request_size=0;
	auto cli = clara::detail::Help(showHelp)
	           | clara::detail::Opt( topology, "topology" )["-t"]["--topology"]("What is the topology type? [ (default) fat_tree | dcell | bcube ]")
	           | clara::detail::Opt( topology_size, "topology size") ["-s"] ["--topology_size"] ("What is the size of the topology? ( default 10 )")
	           | clara::detail::Opt( multicriteria_method, "multicriteria method") ["-m"]["--multicriteria"] ("What is the multicriteria method? [ ahp | ahpg | mcl_ahp | mcl_ahpg | ahp_clusterized | (default) ahpg_clusterized]")
	           | clara::detail::Opt( clustering_method,"clustering method") ["-c"]["--clustering"] ("What is the clustering method? [ (default) mcl | pure_mcl]")
	           | clara::detail::Opt( allocation_type,"Strategy") ["-a"] ["--strategy"] ("What is the allocation strategy? [ (default) naive | dc |  all | pure]")
	           | clara::detail::Opt( start_scheduler_time, "Start Time")["--start_time"]("Start scheduler time")
	           | clara::detail::Opt( end_time, "Finish Time")["--end_time"]("What is the schaduler max time?")
	           // 0 For no Test
	           // 1 For Container Test
	           // 2 For Consolidation Test
	           | clara::detail::Opt( test_type, "Type Test")["--test"]("Which type of test you want?")
	           | clara::detail::Opt( request_size, "Request Size")["--request-size"]("Which is the request size?");
	auto result = cli.parse( clara::detail::Args( argc, argv ) );
	if( !result ) {
		std::cerr << "(gpu_scheduler 45) Error in command line: " << result.errorMessage() <<std::endl;
		exit(1);
	}
	if ( showHelp ) {
		std::cout << cli << std::endl;
		exit(0);
	}
	if( topology != "fat_tree" && topology != "dcell" && topology != "bcube" ) {
		std::cerr << "(gpu_scheduler 53) Invalid entered topology\n";
		exit(0);
	}else{
		options->topology_type=topology;
	}
	if( (topology_size<2 || topology_size>48) && topology_size!=0) {
		std::cerr << "(gpu_scheduler 59) Invalid topology size ( must be between 4 and 48 )\n";
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
	}else if(multicriteria_method=="mcl_ahpg") {
		builder->setAHPG();
		allocation_type="dc";
		options->multicriteria_method="mcl_ahpg";
	}else if(multicriteria_method=="mcl_ahp") {
		builder->setAHP();
		allocation_type="dc";
		options->multicriteria_method="mcl_ahp";
	}else if(multicriteria_method=="ahp_clusterized") {
		builder->setAHP();
		builder->setClusteredAHP();
		options->multicriteria_method="ahp";
		allocation_type="clusterized";
	}else if(multicriteria_method=="ahpg_clusterized") {
		builder->setAHPG();
		builder->setClusteredAHPG();
		options->multicriteria_method="ahpg";
		allocation_type="clusterized";
	}else{
		std::cerr << "(gpu_scheduler 89) Invalid multicriteria method\n";
		exit(0);
	}
	if( clustering_method == "mcl" ) {
		builder->setMCL();
		options->clustering_method=clustering_method;
	}else if( clustering_method == "pure_mcl") {
		builder->setMCL();
		options->clustering_method=clustering_method;
		allocation_type=Allocation_t::PURE;
	}else{
		std::cerr << "(gpu_scheduler 100) Invalid clustering method\n";
		exit(0);
	}
	if( start_scheduler_time < 0 ) {
		std::cerr << "(gpu_scheduler 104) Invalid start scheduler time\n";
		exit(0);
	}else{
		options->start_time=start_scheduler_time;
	}
	if( end_time< -1 ) {
		std::cerr << "(gpu_scheduler 110) Invalid end scheduler time\n";
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
	}else if(allocation_type=="pure") {
		options->allocation_type=Allocation_t::PURE;
	}else if(allocation_type=="clusterized") {
		options->allocation_type=Allocation_t::CLUSTERIZED;
	}else{
		std::cerr << "(gpu_scheduler 126) Invalid allocation type\n";
		exit(0);
	}
	if(test_type >=0 && test_type<=2) {
		options->test_type=test_type;
	}else{
		std::cerr << "(gpu_scheduler 132) Invalid Type of test: " << test_type << "\n";
		exit(0);
	}
	if(request_size<=0 && request_size>=22) {
		std::cerr << "(gpu_scheduler 140) Invalid Size of Request\n";
		exit(0);
	}else{
		options->request_size=request_size;
	}
	options->current_time=options->start_time;
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
	obj.fragmentation = ObjectiveFunction::fragmentation(consumed, total);

	// printf("vcpu footprint\n");
	obj.vcpu_footprint = ObjectiveFunction::vcpu_footprint(consumed, total);

	// printf("ram footprint\n");
	obj.ram_footprint = ObjectiveFunction::ram_footprint(consumed, total);

	// printf("footprint\n");
	obj.footprint = ObjectiveFunction::footprint(consumed, total);
	return obj;
}

inline void delete_tasks(scheduler_t* scheduler, Builder* builder, options_t* options, consumed_resource_t* consumed){
	bool free_success=false;
	// printf("%s #### %s\n", scheduler->allocated_task[0].c_str(),scheduler->allocated_task[1].c_str());
	Container* current;
	Host* temp;
	while(true) {
		if(scheduler->containers_to_delete.empty()) {
			// printf("No containers in the Delete Queue\n");
			break;
		}
		current=scheduler->containers_to_delete.top();
		if(current->getDuration()+current->getAllocatedTime() != options->current_time) {
			// printf("No containers to free\n");
			break;
		}
		scheduler->containers_to_delete.pop();
		if(scheduler->allocated_task [ current->getId() ]!=NULL) {
			temp =  builder->getHost ( scheduler->allocated_task [ current->getId() ] );
		}else{
			continue;
		}
		// printf("Scheduler Time %d Deleting container %d\n", options->current_time, current->getId());
		const bool active_status=temp->getActive();
		free_success=Allocator::freeHostResource(
			/* the specific host that have the container*/
			temp,
			/* The container to be removed*/
			current,
			/* The consumed DC status*/
			consumed
			);
		if(!free_success) {
			std::cerr << "(gpu_scheduler 200) gpu_scheduler(170) - Error in free the task " << current->getId() << " from the data center\n";
			exit(1);
		}
		if(temp->getAllocatedResources()==0) {
			temp->setActive(false);
			//Check if the server was on and now is off
			if(active_status==true) {
				consumed->active_servers--;
			}
		}

		// Search the container C in the vector and removes it
		scheduler->allocated_task.erase(current->getId());

		delete(current);
	}
}

inline void allocate_tasks(scheduler_t* scheduler, Builder* builder, options_t* options, consumed_resource_t* consumed){
	bool allocation_success=false;
	// Check the task submission
	Container* current;

	int total_delay=0;
	while(true) {
		if(scheduler->containers_to_allocate.empty()) {
			break;
		}
		current = scheduler->containers_to_allocate.top();
		if( current->getSubmission()+current->getDelay() != options->current_time) {
			// printf("Scheduler Time %d and Container %d Time %d\n", options->current_time, current->getId(),current->getSubmission()+current->getDelay());
			break;
		}
		// printf("\tREMOVE ALLOCATING CONTAINER %d\n", current->getId());
		scheduler->containers_to_allocate.pop();
		// allocate the new task in the data center.
		if( options->allocation_type==Allocation_t::PURE) {
			allocation_success=Allocator::mcl_pure(builder,current,scheduler->allocated_task);
		} else if( options->allocation_type == Allocation_t::NAIVE) {
			allocation_success=Allocator::naive(builder,current, scheduler->allocated_task,consumed);
		}else if( options->allocation_type == Allocation_t::DC) {
			allocation_success=Allocator::dc(builder,current,scheduler->allocated_task,consumed);
		} else if ( options->allocation_type == Allocation_t::ALL) {
			allocation_success=Allocator::all();
		} else if ( options->allocation_type == Allocation_t::CLUSTERIZED) {
			allocation_success=Allocator::ahp_clusterized(builder, current, scheduler->allocated_task,consumed);
		} else {
			std::cerr << "(gpu_scheduler 249) Invalid type\n";
			exit(1);
		}
		if(!allocation_success) {
			if(current->getResource()->vcpu_max>24 || current->getResource()->ram_max>256) {
				delete(current);
				continue;
			}
			printf("\tContainer %d Add Delay, old time %d\n", current->getId(), current->getSubmission()+current->getDelay());
			int delay=1;
			if(!scheduler->containers_to_delete.empty()) {
				Container* first_to_delete = scheduler->containers_to_delete.top();
				delay = (first_to_delete->getDuration() + first_to_delete->getAllocatedTime()) - ( current->getSubmission() + current->getDelay() ) +1;
			}
			current->addDelay(delay);
			// printf("delay,%d,%d\n",current->getId(), current->getDelay());

			scheduler->containers_to_allocate.push(current);

			total_delay+=current->getDelay();
		}else{
			printf("\tContainer %d Allocated in time %d\n", current->getId(), current->getSubmission()+current->getDelay() );
			current->setAllocatedTime(options->current_time);

			scheduler->containers_to_delete.push(current);
			// scheduler->containers_to_allocate.pop();
		}
	}
	printf("total_delay,%d,%d\n", options->current_time,total_delay);
}

void schedule(Builder* builder, Comunicator* conn, scheduler_t* scheduler, options_t* options, int message_count){
	const int total_containers = scheduler->containers_to_allocate.size();
	//Create the variable to store all the data center resource
	total_resources_t total_resources;
	// printf("Setting DC resources\n");
	builder->setDataCenterResources(&total_resources);
	// printf("set\n");
	consumed_resource_t consumed_resources;
	consumed_resources.vcpu=0;
	consumed_resources.ram=0;
	consumed_resources.active_servers=0;
	consumed_resources.time=0;

	objective_function_t objective;
	objective.time=0;
	objective.fragmentation=0;
	objective.footprint=0;
	objective.vcpu_footprint=0;
	objective.ram_footprint=0;

	while(message_count>0 || options->current_time <= options->end_time || !scheduler->containers_to_allocate.empty() || !scheduler->containers_to_delete.empty()) {
		consumed_resources.time=options->current_time;
		// #endif
		// if(options->current_time==options->end_time) break;
		// std::cout<<"Scheduler Time "<< options->current_time<<"\n";
		// std::cout<<"message_count "<<message_count<<"\n";
		// make sure there is work in the queue
		if(message_count>0) {
			while(true) {
				// Create new container
				Container *current = new Container();
				// Set the resources to the container
				current->setTask(conn->getNextTask());
				// Put the container in the vector
				scheduler->containers_to_allocate.push(current);
				// getchar();
				message_count--;
				// printf("Receiving new container %d\n in time %d", current->getId(), options->current_time);
				if(current->getSubmission()!=options->current_time) {
					break;
				}
			}
			// Print the container
			// std::cout << *c << "\n";
		}
		// Search the containers to delete
		delete_tasks(scheduler, builder, options, &consumed_resources);
		// Search the containers in the vector to allocate in the DC
		allocate_tasks(scheduler, builder, options, &consumed_resources);
		// Update the lifetime
		// getchar();
		// consumed_resources.active_servers = builder->getTotalActiveHosts();
		objective=calculateObjectiveFunction(consumed_resources, total_resources);
		if(options->test_type==2) {
			printf("%d,%.7lf,%.7lf,%.7lf,%.7lf,%.5lf%%\n",
			       objective.time,
			       objective.fragmentation,
			       objective.footprint,
			       objective.vcpu_footprint,
			       objective.ram_footprint,
			       (100-(100.0*(scheduler->containers_to_allocate.size()/(float)total_containers)))
			       );
		}
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
	setup(argc,argv,builder,&scheduler,&options);

	if (options.test_type==0) {     // no test is set
		// Creating the communicatior
		Comunicator *conn = new Comunicator();
		conn->setup();
		int message_count=conn->getQueueSize();

		schedule(builder, conn, &scheduler, &options, message_count);

		delete(conn);

	}else if(options.test_type==1 || options.test_type==2) {
		// parse all json
		Reader* reader = new Reader();
		std::string path = "../simulator/json/datacenter/google-";
		path+= std::to_string(options.request_size);
		path+=".json";
		reader->openDocument(path.c_str());
		std::string message;
		while((message=reader->getNextTask())!="eof") {
			// Create new container
			Container *current = new Container();
			// Set the resources to the container
			current->setTask(message.c_str());
			// Put the container in the vector
			scheduler.containers_to_allocate.push(current);
		}
		delete(reader);
		builder->runClustering(builder->getHosts());
		builder->getClusteringResult();
		// Scalability Test or Objective Function Test
		// force cout to not print in cientific notation
		std::cout<<std::fixed;

		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		schedule(builder, NULL, &scheduler, &options, 0);

		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);

		std::cout<<options.multicriteria_method<<";" << options.topology_size << ";" << time_span.count() << "\n";
	}

	// Free the allocated pointers
	delete(builder);
	return 0;
}
