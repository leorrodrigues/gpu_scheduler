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

	// if(this->path!=NULL) {
	//      for(size_t i=0; i<this->links_size; i++) {
	//              free(this->path[i]);
	//              free(this->path_edge[i]);
	//      }
	//      free(this->path);
	//      free(this->path_edge);
	//      free(this->destination);
	//      free(this->values);
	//      this->path=NULL;
	//      this->path_edge=NULL;
	//      this->destination=NULL;
	//      this->values=NULL;
	// }
}

void Task::setTask(const char* taskMessage){
	rapidjson::Document task;
	task.Parse(taskMessage);

	/**********************************************************/
	/********                   Task variables              *********/
	/*********************************************************/
	spdlog::debug("Getting the task variables");
	//Duration
	this->duration = task["duration"].GetFloat();

	//ID
	this->id = task["id"].GetInt();

	//SUBMISSION
	this->submission = task["submission"].GetFloat();

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

		for(size_t i=0; i < this->containers_size; i++) {
			spdlog::debug("Creating a new container");
			c = new Container();
			//CRIA UM CONTAINER E COLOCA OS VALORES DENTRO DELE, APOS ISSO ADICIONA ELE DENTRO DO ARRAY DE CONTAINERS =).

			// for(size_t r_s=0; r_s < this->resources.size(); r_s++) {
			spdlog::debug("Reading the information of this container");
			for(auto [key, val] : this->resources) {
				spdlog::debug("Gettin the values of {}", key);

				std::string min  = key+"_min";
				std::string max = key+"_max";

				if(!containersArray[i].HasMember(max.c_str())) continue;

				float vmin  = containersArray[i][max.c_str()].GetFloat();
				float vmax = containersArray[i][min.c_str()].GetFloat();

				c->setValue(key, vmin, false);
				c->setValue(key, vmax, true);
				this->setValue(key, vmin, false);
				this->setValue(key, vmax, true);
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

		this->pods_size = temp_pods.size();
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
		for(size_t i=0; i < this->links_size; i++) {
			spdlog::debug("\tGetting the link {}",i);
			source = linksArray[i]["source"].GetInt();
			spdlog::debug("\tsource get");
			//Finding the respective container
			this->containers[source-1]->setLink(
				linksArray[i]["destination"].GetInt(),
				linksArray[i]["bandwidth_min"].GetFloat(),
				linksArray[i]["bandwidth_max"].GetFloat()
				);
			spdlog::debug("\tlink set");
		}
	}
	for(size_t i=0; i<pods_size; i++) {
		spdlog::debug("Update de Bandwidth of the pod {}",i);
		this->pods[i]->updateBandwidth();
		spdlog::debug("\tUpdated");
	}
}

void Task::addDelay(){
	this->delay++;
}

void Task::addDelay(unsigned int delay){
	this->delay+=delay;
}

void Task::setAllocatedTime(unsigned int allocatedTime){
	this->allocated_time = allocatedTime;
}

void Task::setSubmission(unsigned int submission){
	this->submission = submission;
}

void Task::setLinkPath(int** path){
	this->path=path;
}

void Task::setLinkPathEdge(int** path_edge){
	this->path_edge=path_edge;
}

void Task::setLinkDestination(int* destination){
	this->destination=destination;
}

void Task::setLinkValues(float* values){
	this->values=values;
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

int** Task::getLinkPath(){
	return this->path;
}

int** Task::getLinkPathEdge(){
	return this->path_edge;
}

int* Task::getLinkDestination(){
	return this->destination;
}

float* Task::getLinkValues(){
	return this->values;
}

void Task::print() {
	spdlog::debug("Task:{");
	spdlog::debug("Duration {}",this->duration);
	spdlog::debug("\tPods:[");
	for(size_t i=0; i<this->pods_size; i++) {
		this->pods[i]->print();
	}
	spdlog::debug("\t]");
	spdlog::debug("\tId: {}",this->id);
	spdlog::debug("\tSubmission: {}",this->submission);
	spdlog::debug("\tTotal Resources");
	for(auto const& [key, val] : this->resources) {
		spdlog::debug("\t\t\t{} - {}; {}; {}",key, std::get<0>(val), std::get<1>(val),  std::get<2>(val));
	}
	spdlog::debug("}");
}
