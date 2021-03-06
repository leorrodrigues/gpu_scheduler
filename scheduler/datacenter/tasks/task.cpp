#include "task.hpp"

Task::Task() : Task_Resources(){
	this->pods=NULL;
	this->containers=NULL;
	this->duration=0;
	this->submission=0;

	this->pods_size = 0;
	this->containers_size=0;
	this->links_size=0;
	this->allocated_time=0;
	this->delay=0;

	this->path = NULL;
	this->path_edge = NULL;
	this->destination = NULL;
	this->values = NULL;
}

Task::~Task(){
	for(size_t i=0; i<this->pods_size; i++)
		delete(this->pods[i]);
	free(this->pods);
	this->pods=NULL;

	free(containers); // dont free each container (containers[i]), because it is freed by cascade when you call delete(pod). Free only the array of pointers.
	this->containers=NULL;
}

void Task::setTask(const char* taskMessage){
	rapidjson::Document task;
	task.Parse(taskMessage);

	/**********************************************************/
	/********                   Task variables              *********/
	/*********************************************************/
	spdlog::debug("Getting the task variables");
	//Duration
	this->duration = static_cast<unsigned int>(task["duration"].GetFloat());
	if(this->duration==0) ++this->duration;
	//Early Duration
	if(task.HasMember("early")) {
		this->early_duration = static_cast<unsigned int>(task["early"].GetFloat());
	} else {
		this->early_duration = 0;
	}
	//ID
	this->id = task["id"].GetInt();

	//SUBMISSION
	this->submission = static_cast<unsigned int>(task["submission"].GetFloat());

	if(task.HasMember("deadline")) {
		this->deadline = static_cast<unsigned int>(task["deadline"].GetFloat());
	}else{
		this->deadline = 0;
	}
	/**********************************************************/
	/********              Pod   variables                    ********/
	/*********************************************************/
	spdlog::debug("Getting the pod variables");
	const rapidjson::Value &containersArray = task["containers"];
	this->containers_size = containersArray.Size();
	this->containers = NULL;

	if(this->containers_size!=0) {
		std::map<unsigned int, Pod*> temp_pods;
		Pod* pod = NULL;
		this->containers=(Container**) malloc (sizeof(Container*)*this->containers_size);
		Container *c=NULL;

		for(unsigned int i=0; i < this->containers_size; i++) {
			spdlog::debug("Creating a new container");
			c = new Container();
			// Creates a container and insert the values inside it. After, it'll be added into the container's array.

			spdlog::debug("Reading the information of this container");
			for(auto [key, val] : this->resources) {
				spdlog::debug("Gettin the values of {}", key);

				std::string min  = key+"_min";
				std::string max = key+"_max";

				if(!containersArray[i].HasMember(max.c_str())) continue;

				float vmax  = containersArray[i][max.c_str()].GetFloat();
				float vmin = containersArray[i][min.c_str()].GetFloat();

				c->setValue(key, vmin, false);
				c->setValue(key, vmax, true);
				this->addValue(key, vmin, false);
				this->addValue(key, vmax, true);
			}
			//POD
			int pod_index = containersArray[i]["pod"].GetInt();
			// The pod isn't already created, create a new pod and then set the container into it.
			spdlog::debug("Searching if the pod exists");
			if(temp_pods.find(pod_index) == temp_pods.end()) {
				pod = new Pod(pod_index);
				temp_pods[pod_index] = pod;
			}
			//NAME
			spdlog::debug("Adding the container into the pod");
			c->setId(containersArray[i]["name"].GetInt());

			temp_pods[pod_index]->addContainer(c);
			this->containers[i]=c;
		}
		spdlog::debug("Updating the last variables of the pod");
		bool s_z=false;
		if(temp_pods.find(0) != temp_pods.end()) {
			s_z=true;
		}

		this->pods_size = static_cast<unsigned int>(temp_pods.size());
		//Create the pods array
		this->pods = (Pod**) malloc (sizeof(Pod*)*this->pods_size);

		//Populate the pods array
		spdlog::debug("Populating all the pods in the task array");
		if(s_z) {
			for(
				std::map<unsigned int,Pod*>::iterator it=temp_pods.begin();
				it!=temp_pods.end();
				it++
				) {
				this->pods[it->first] = it->second;
			}
		}else{
			for(
				std::map<unsigned int,Pod*>::iterator it=temp_pods.begin();
				it!=temp_pods.end();
				it++
				) {
				this->pods[it->first-1] = it->second;
				this->pods[it->first-1]->subId();
			}
		}
	}
	//Now we have all te pods constructed and their respectives containers inside it.
	//To construct the links between two containers, use the containers array to easy do this job. As the elements in the pods and in this vector are the pointers to the same memory address.

	/**********************************************************/
	/********                   Link variables                *********/
	/*********************************************************/
	//Links
	spdlog::debug("Getting the link variables");
	const rapidjson::Value &linksArray = task["links"];
	this->links_size = linksArray.Size();
	unsigned int source=0;
	if(this->links_size!=0) {
		//Iterating through all the links in the request
		for(unsigned int i=0; i < this->links_size; i++) {
			spdlog::debug("\tGetting the link {}",i);
			source = linksArray[i]["source"].GetInt();
			spdlog::debug("\tsource get");
			//Finding the respective container
			this->containers[source-1]->setLink(
				linksArray[i]["destination"].GetInt(),
				linksArray[i]["bandwidth_min"].GetFloat(),
				linksArray[i]["bandwidth_max"].GetFloat()
				);
			// update the max bandwidth for the task
			this->addValue("bandwidth", linksArray[i]["bandwidth_min"].GetFloat(),false);
			this->addValue("bandwidth", linksArray[i]["bandwidth_max"].GetFloat(),true);
			spdlog::debug("\tlink set");
		}
	}
	for(size_t i=0; i<pods_size; i++) {
		spdlog::debug("Update de Bandwidth of the pod {}",i);
		this->pods[i]->updateBandwidth();
		spdlog::debug("\tUpdated");
	}
}

void Task::setRequestedTime(){
	this->requested_time = std::chrono::high_resolution_clock::now();
}

void Task::setStartTime(){
	this->start_time = std::chrono::high_resolution_clock::now();
}

void Task::setStopTime(){
	this->stop_time = std::chrono::high_resolution_clock::now();
}

void Task::addDelay(unsigned int delay){
	this->delay += delay;
}

void Task::setAllocatedTime(unsigned int allocatedTime){
	this->allocated_time = allocatedTime;
}

void Task::setSubmission(unsigned int submission){
	this->submission = submission;
}

void Task::setLinkPath(int* path){
	this->path=path;
}

void Task::setLinkPathEdge(int* path_edge){
	this->path_edge=path_edge;
}

void Task::setLinkDestination(int* destination){
	this->destination=destination;
}

void Task::setLinkInit(int* init){
	this->init=init;
}

void Task::setLinkValues(float* values){
	this->values=values;
}

std::chrono::high_resolution_clock::time_point Task::getRequestedTime(){
	return this->requested_time;
}

std::chrono::high_resolution_clock::time_point Task::getStartTime(){
	return this->start_time;
}

std::chrono::high_resolution_clock::time_point Task::getStopTime(){
	return this->stop_time;
}

Pod** Task::getPods(){
	return this->pods;
}

Container** Task::getContainers(){
	return this->containers;
}

unsigned int Task::getPodsSize(){
	return this->pods_size;
}

unsigned int Task::getLinksSize(){
	return this->links_size;
}

unsigned int Task::getDuration(){
	return this->duration;
}

unsigned int Task::getEarlyDuration(){
	return this->early_duration;
}

unsigned int Task::getSubmission(){
	return this->submission;
}

unsigned int Task::getContainersSize(){
	return this->containers_size;
}

unsigned int Task::getAllocatedTime(){
	return this->allocated_time;
}

unsigned int Task::getDelay(){
	return this->delay;
}

unsigned int Task::getDeadline(){
	return this->deadline;
}

int* Task::getLinkPath(){
	return this->path;
}

int* Task::getLinkPathEdge(){
	return this->path_edge;
}

int* Task::getLinkDestination(){
	return this->destination;
}

int* Task::getLinkInit(){
	return this->init;
}

float* Task::getLinkValues(){
	return this->values;
}

float Task::getBandwidthAllocated(){
	float allocated=0;

	for(size_t i=0; i<pods_size; i++) {
		allocated+=pods[i]->getTotalAllocated("bandwidth");
	}
	return allocated;
}

void Task::print() {
	spdlog::debug("Task:{");
	spdlog::debug("\tId: {}",this->id);
	spdlog::debug("\tSubmission: {}",this->submission);
	spdlog::debug("\tDuration {}",this->duration);
	spdlog::debug("\tDeadline {}",this->deadline);
	spdlog::debug("\tPods:[");
	for(size_t i=0; i<this->pods_size; i++) {
		this->pods[i]->print();
	}
	spdlog::debug("\t]");
	spdlog::debug("\tTotal Resources");
	for(auto const& [key, val] : this->resources) {
		spdlog::debug("\t\t\t{} - {}; {}; {}",key, val[0], val[1], val[2]);
	}
	spdlog::debug("}");
}

float Task::taskUtility(){
	float max=0, allocated=0;

	for(auto const&it : this->total_allocated) {
		for(size_t i=0; i<pods_size; i++) {
			allocated+=pods[i]->getTotalAllocated(it.first);
			max+=pods[i]->getMaxResource(it.first);
		}
	}
	if(max<allocated)
		SPDLOG_ERROR("DC UTILITY MAX < ALLOCATED");
	return max!=0 ? allocated/max : 0;
}

float Task::linkUtility(){
	float max=0, allocated=0;

	for(size_t i=0; i<pods_size; i++) {
		allocated+=pods[i]->getTotalAllocated("bandwidth");
		max+=pods[i]->getMaxResource("bandwidth");
	}
	if(max<allocated) {
		SPDLOG_ERROR("LINK UTILITY MAX < ALLOCATED");
	}
	return max!=0 ? allocated/max : 0;
}
