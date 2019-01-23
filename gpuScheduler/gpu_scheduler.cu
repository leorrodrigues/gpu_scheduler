#include <iostream>
#include <chrono>
#include <ctime>
#include <ratio>

#include "types.hpp"
#include "reader.hpp"
#include "builder.cuh"
#include "thirdparty/clara.hpp"

#include "allocator/multicriteria_clusterized.cuh"
#include "allocator/pure_mcl.hpp"
#include "allocator/naive.hpp"
#include "allocator/utils.hpp"
#include "allocator/free.hpp"
#include "allocator/all.cuh"

#include "objective_functions/fragmentation.hpp"
#include "objective_functions/footprint.hpp"

void setup(int argc, char** argv, Builder* builder, scheduler_t *scheduler, options_t* options){
	std::string topology = "fat_tree";
	std::string multicriteria_method = "ahpg_clusterized";
	std::string clustering_method = "mcl";
	int topology_size=10;
	// dont show the help by default. Use `-h or `--help` to enable it.
	bool showHelp = false;
	int test_type=0;
	int request_size=0;
	auto cli = clara::detail::Help(showHelp)
	           | clara::detail::Opt( topology, "topology" )["-t"]["--topology"]("What is the topology type? [ (default) fat_tree | dcell | bcube ]")
	           | clara::detail::Opt( topology_size, "topology size") ["-s"] ["--topology_size"] ("What is the size of the topology? ( default 10 )")
	           | clara::detail::Opt( multicriteria_method, "multicriteria method") ["-m"]["--multicriteria"] ("What is the multicriteria method? [ ahp | (default) ahpg | topsis]")
	           | clara::detail::Opt( clustering_method,"clustering method") ["-c"]["--clustering"] ("What is the clustering method? [ (default) mcl | pure_mcl | none ]")
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
		std::cerr << "(gpu_scheduler 80) Invalid clustering method\n";
		exit(0);
	}
	options->clustering_method=clustering_method;

	if(!cluster) {
		if( multicriteria_method == "ahpg") {
			builder->setAHPG();
		}else if(multicriteria_method == "ahp" ) {
			builder->setAHP();
		}else if(multicriteria_method=="topsis") {
			builder->setTOPSIS();
		} else{
			std::cerr << "(gpu_scheduler 92) Invalid multicriteria method\n";
			exit(0);
		}
		options->multicriteria_method=multicriteria_method;
	}else{
		if(multicriteria_method=="ahp") {
			builder->setClusteredAHP();
		}else if(multicriteria_method=="ahpg") {
			builder->setClusteredAHPG();
		}else if(multicriteria_method=="topsis") {
			builder->setClusteredTOPSIS();
		}else{
			std::cerr << "(gpu_scheduler 93) Invalid multicriteria method\n";
			exit(0);
		}
	}

	if(test_type >=0 && test_type<=3) {
		options->test_type=test_type;
	}else{
		std::cerr << "(gpu_scheduler 136) Invalid Type of test: " << test_type << "\n";
		exit(0);
	}

	if(request_size<=0 && request_size>=22) {
		std::cerr << "(gpu_scheduler 140) Invalid Size of Request\n";
		exit(0);
	}else{
		options->request_size=request_size;
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

	bool free_success = false;
	Container* current = NULL;
	Host* temp = NULL;

	while(true) {

		if(scheduler->containers_to_delete.empty()) {
			break;
		}

		current=scheduler->containers_to_delete.top();

		if( current->getDuration() + current->getAllocatedTime() != options->current_time) {
			break;
		}

		scheduler->containers_to_delete.pop();

		// if( scheduler->allocated_task [ current->getId() ]!=NULL ) {
		temp =  builder->getHost ( scheduler->allocated_task [ current->getId() ] );
		// } else {
		// continue;
		// }

		printf("Scheduler Time %d Deleting container %d\n", options->current_time, current->getId());

		const bool active_status=temp->getActive();

		free_success=Allocator::freeHostResource(
			/* the specific host that have the container*/
			temp,
			/* The container to be removed*/
			current,
			/* The consumed DC status*/
			consumed,
			builder
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

		// Search the container in the vector and removes it
		scheduler->allocated_task.erase(current->getId());

		delete(current);
	}

	current = NULL;
	temp = NULL;
}

inline void allocate_tasks(scheduler_t* scheduler, Builder* builder, options_t* options, consumed_resource_t* consumed, total_resources_t* total_dc){

	bool allocation_success = false;
	Container* current = NULL;
	int total_delay = 0;
	int delay=1;

	while(true) {
		if(scheduler->containers_to_allocate.empty()) {
			break;
		}

		current = scheduler->containers_to_allocate.top();

		if( current->getSubmission()+current->getDelay() != options->current_time) {
			break;
		}

		scheduler->containers_to_allocate.pop();

		if(Allocator::checkFit(total_dc, consumed,current)!=0) {
			// allocate the new task in the data center.
			if( options->clustering_method=="pure_mcl") {
				allocation_success=Allocator::mcl_pure(builder,current,scheduler->allocated_task);
			} else if( options->clustering_method == "none") {
				printf("Call naive type\n");
				allocation_success=Allocator::naive(builder,current, scheduler->allocated_task,consumed);
			} else if ( options->clustering_method == "mcl") {
				allocation_success=Allocator::multicriteria_clusterized(builder, current, scheduler->allocated_task,consumed);
			} else if ( options->clustering_method == "all") {
				allocation_success=Allocator::all();
			} else {
				std::cerr << "(gpu_scheduler 266) Invalid type\n";
				exit(1);
			}
		}else{
			allocation_success=false;
		}

		if(!allocation_success) {

			if(!scheduler->containers_to_delete.empty()) {
				Container* first_to_delete = scheduler->containers_to_delete.top();

				delay = (first_to_delete->getDuration() + first_to_delete->getAllocatedTime()) - ( current->getSubmission() + current->getDelay() );
			}

			current->addDelay(delay);

			scheduler->containers_to_allocate.push(current);

			total_delay+=current->getDelay();

			printf("\tContainer %d Added Delay in time %d\n", current->getId(), current->getSubmission()+current->getDelay() );

		}else{

			printf("\tContainer %d Allocated in time %d\n", current->getId(), current->getSubmission()+current->getDelay() );
			current->setAllocatedTime(options->current_time);
			scheduler->containers_to_delete.push(current);

		}
	}
	printf("total_delay,%d,%d\n", options->current_time,total_delay);
}

void schedule(Builder* builder, Comunicator* conn, scheduler_t* scheduler, options_t* options, int message_count){
	const int total_containers = scheduler->containers_to_allocate.size();
	//Create the variable to store all the data center resource
	total_resources_t total_resources;

	builder->setDataCenterResources(&total_resources);

	consumed_resource_t consumed_resources;
	objective_function_t objective;

	while(
		!scheduler->containers_to_allocate.empty() ||
		!scheduler->containers_to_delete.empty()
		) {

		consumed_resources.time = options->current_time;
		// if(options->current_time==options->end_time) break;
		// std::cout<<"Scheduler Time "<< options->current_time<<"\n";
		// std::cout<<"message_count "<<message_count<<"\n";

		//************************************************//
		//     READ CONTAINER REQUEST THROUGH RABBITMQ    //
		//************************************************//
		// while(message_count>0 || options->current_time <= options->end_time || !scheduler->containers_to_allocate.empty() || !scheduler->containers_to_delete.empty()) {
		// if(message_count>0) {
		//      while(true) {
		//              // Create new container
		//              Container *current = new Container();
		//              // Set the resources to the container
		//              current->setTask(conn->getNextTask());
		//              // Put the container in the vector
		//              scheduler->containers_to_allocate.push(current);
		//              // getchar();
		//              message_count--;
		//              // printf("Receiving new container %d\n in time %d", current->getId(), options->current_time);
		//              if(current->getSubmission()!=options->current_time) {
		//                      break;
		//              }
		//      }
		//    // Print the container
		//    std::cout << *c << "\n";
		// }
		//************************************************//
		//************************************************//
		//************************************************//

		// Search the containers to delete
		delete_tasks(scheduler, builder, options, &consumed_resources);
		// Search the containers in the vector to allocate in the DC
		allocate_tasks(scheduler, builder, options, &consumed_resources, &total_resources);

		//************************************************//
		//       Print All the metrics information        //
		//************************************************//
		objective=calculateObjectiveFunction(consumed_resources, total_resources);

		if(options->test_type==2) {
			printf("%d,%.7lf,%.7lf,%.7lf,%.7lf,%.5lf%%\n",
			       objective.time,
			       objective.fragmentation,
			       objective.footprint,
			       objective.vcpu_footprint,
			       objective.ram_footprint,
			       (100 - (
					100.0*(
						scheduler->containers_to_allocate.size()/(float)total_containers)
					)
			       )
			       );
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
	setup(argc,argv,builder,&scheduler,&options);

	if (options.test_type==0) {     // no test is set
		// Creating the communicatior
		Comunicator *conn = new Comunicator();
		conn->setup();
		int message_count=conn->getQueueSize();

		schedule(builder, conn, &scheduler, &options, message_count);

		delete(conn);

	}else{
		// parse all json
		Reader* reader = new Reader();
		std::string path = "../simulator/json/";
		if(options.test_type==1) {
			path+="container/data-";
		}
		else if(options.test_type==2) {
			path+="datacenter/google-";
		}
		else if(options.test_type==3) {
			path+="datacenter/data-";
		}
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
		message.clear();
		delete(reader);
		if(options.clustering_method!="none") {
			builder->runClustering(builder->getHosts());

			builder->getClusteringResult();
		}
		// Scalability Test or Objective Function Test
		// force cout to not print in cientific notation
		std::cout<<std::fixed;

		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		printf("Calling the scheduler\n");
		schedule(builder, NULL, &scheduler, &options, 0);

		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);

		std::cout<<options.multicriteria_method<<";" << options.topology_size << ";" << time_span.count() << "\n";
	}

	// Free the allocated pointers
	delete(builder);
	return 0;
}
