#include "task.hpp"

Task::Task(){
	this->pods=NULL;
	this->duration=0;
	this->submission=0;

	this->pods_size = 0;
	this->allocated_time=0;
	this->delay=0;
	this->fit=0;
}

Task::~Task(){
	for(size_t i=0; i<this->pods_size; i++)
		delete(this->pods[i]);
	free(this->pods);
	pods=NULL;
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
	Container **containers = NULL;
	if(this->containers_size!=0) {

		std::map<unsigned int, Pod*> temp_pods;
		Pod* pod = NULL;
		containers=(Container**) malloc (sizeof(Container*)*this->containers_size);
		Container *c=NULL;

		const char *resources[]={
			"epc_min","epc_max",
			"ram_min","ram_max",
			"vcpu_min","vcpu_max"
		};
		unsigned int resources_size=6;

		for(size_t i=0; i < this->containers_size; i++) {
			c = new Container();
			//CRIA UM CONTAINER E COLOCA OS VALORES DENTRO DELE, APOS ISSO ADICIONA ELE DENTRO DO ARRAY DE CONTAINERS =).

			for(size_t r_s=0; r_s < resources_size; r_s++) {
				c->setValue(
					resources[r_s],
					containersArray[i][resources[r_s]].GetDouble()
					);
				this->resources[resources[r_s]]+=c->getResource(resources[r_s]);
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
			containers[i]=c;
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
			for(size_t j=0; j< this->containers_size; j++) {
				if(containers[j]->getId()==source) {
					containers[j]->setLink(
						linksArray[i]["destination"].GetInt(),
						linksArray[i]["bandwidth_min"].GetDouble(),
						linksArray[i]["bandwidth_max"].GetDouble()
						);
				}
			}
		}
	}

	free(containers);
}

void Task::addDelay(){
	this->delay++;
}

void Task::addDelay(unsigned int delay){
	this->delay+=delay;
}

void Task::setFit(unsigned int fit){
	this->fit=fit;
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

unsigned int Task::getPodsSize(){
	return this->pods_size;
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

unsigned int Task::getFit(){
	return this->fit;
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
	os<<"\t\tepc min: " <<t.resources.at("epc_min")<< "; epc_max: " <<t.resources.at("epc_max")<<"\n";
	os<<"\t\tvcpu min: "<<t.resources.at("vcpu_min")<<"; vcpu_max: "<<t.resources.at("vcpu_max")<<"\n";
	os<<"\t\tram min: " <<t.resources.at("ram_min")<< "; ram_max: " <<t.resources.at("ram_max")<<"\n";
	os<<"}\n";
	return os;
}
