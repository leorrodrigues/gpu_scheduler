#include "task.hpp"

Task::Task() : Task_Resources(){
	this->pods=NULL;
	this->containers=NULL;
	this->duration=0;
	this->submission=0;

	this->pods_size = 0;
	this->allocated_time=0;
	this->delay=0;
}

Task::~Task(){
	for(size_t i=0; i<this->pods_size; i++)
		delete(this->pods[i]);
	free(this->pods);
	this->pods=NULL;
	this->containers=NULL;
}

void Task::setTask(const char* taskMessage){
	rapidjson::Document task;
	task.Parse(taskMessage);

	/**********************************************************/
	/********                   Task variables              *********/
	/*********************************************************/
	//Duration
	this->duration = task["duration"].GetDouble();

	//ID
	this->id = task["id"].GetInt();

	//SUBMISSION
	this->submission = task["submission"].GetDouble();

	/**********************************************************/
	/********              Pod   variables                    ********/
	/*********************************************************/
	const rapidjson::Value &containersArray = task["containers"];
	this->containers_size = containersArray.Size();
	this->containers = NULL;

	if(this->containers_size!=0) {
		std::map<unsigned int, Pod*> temp_pods;
		Pod* pod = NULL;
		this->containers=(Container**) malloc (sizeof(Container*)*this->containers_size);
		Container *c=NULL;

		for(size_t i=0; i < this->containers_size; i++) {
			c = new Container();
			//CRIA UM CONTAINER E COLOCA OS VALORES DENTRO DELE, APOS ISSO ADICIONA ELE DENTRO DO ARRAY DE CONTAINERS =).

			// for(size_t r_s=0; r_s < this->resources.size(); r_s++) {
			for(auto [key, val] : this->resources) {
				std::string min  = key+"_min";
				std::string max = key+"_max";

				if(!containersArray[i].HasMember(max.c_str())) continue;

				float vmin  = containersArray[i][max.c_str()].GetDouble();
				float vmax = containersArray[i][min.c_str()].GetDouble();

				c->setValue(key, vmin, false);
				c->setValue(key, vmax, true);
				this->setValue(key, vmin, false);
				this->setValue(key, vmax, true);
			}
			//POD
			int pod_index = containersArray[i]["pod"].GetInt();

			// The pod isn't already created, create a new pod and then set the container into it.
			if(temp_pods.find(pod_index) == temp_pods.end()) {
				pod = new Pod(pod_index);
				temp_pods[pod_index] = pod;
			}
			//NAME
			c->setId(containersArray[i]["name"].GetInt());

			temp_pods[pod_index]->addContainer(c);
			this->containers[i]=c;
		}
		this->pods_size = temp_pods.size();
		//Create the pods array
		this->pods = (Pod**) malloc (sizeof(Pod*)*this->pods_size);

		//Populate the pods array
		for(
			std::map<unsigned int,Pod*>::iterator it=temp_pods.begin();
			it!=temp_pods.end();
			it++
			) {
			this->pods[it->first-1] = it->second;
		}
	}
	//Now we have all te pods constructed and their respectives containers inside it.
	//To construct the links between two containers, use the containers array to easy do this job. As the elements in the pods and in this vector are the pointers to the same memory address.

	/**********************************************************/
	/********                   Link variables                *********/
	/*********************************************************/
	//Links
	const rapidjson::Value &linksArray = task["links"];
	this->links_size = linksArray.Size();
	unsigned int source=0;
	if(this->links_size!=0) {
		//Iterating through all the links in the request
		for(size_t i=0; i < this->links_size; i++) {
			source = linksArray[i]["source"].GetInt();
			//Finding the respective container
			this->containers[source-1]->setLink(
				linksArray[i]["destination"].GetInt(),
				linksArray[i]["bandwidth_min"].GetDouble(),
				linksArray[i]["bandwidth_max"].GetDouble()
				);
		}
	}
	for(size_t i=0; i<pods_size; i++) {
		this->pods[i]->updateBandwidth();
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

std::ostream& operator<<(std::ostream& os, const Task& t)  {
	os<<"Task:{\n";
	os<<"\tDuration: "<<t.duration<<"\n";
	os<<"\tPods:[\n";
	for(size_t i=0; i<t.pods_size; i++) {
		os<<(*t.pods[i]);
	}
	os<<"\t]\n";
	os<<"\tId: "<< t.id<<"\n";
	os<<"\tSubmission: "<<t.submission<<"\n";
	os<<"\tTotal Resources\n";
	for(auto const& [key, val] : t.resources) {
		os<<"\t\t\t"<<key<<"-";
		os<<std::get<0>(val)<<";";
		os<<std::get<1>(val)<<";";
		os<<std::get<2>(val);
		os<<"\n";
	}
	os<<"}\n";
	return os;
}
