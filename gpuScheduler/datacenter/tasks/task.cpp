#include "task.hpp"

Task::Task(){
	this->pods=NULL;
	this->duration=0;
	this->id=0;
	this->submission=0;

	this->pods_size = 0;
	this->allocated_time=0;
	this->delay=0;
	this->fit=0;

	this->epc_min=0;
	this->ram_min=0;
	this->vcpu_min=0;
	this->epc_max=0;
	this->ram_max=0;
	this->vcpu_max=0;
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
	/********                   Link variables                *********/
	/*********************************************************/

	//Links
	const rapidjson::Value &linksArray = task["links"];
	unsigned int links_size = linksArray.Size();
	Link** links=NULL;
	if(links_size!=0) {
		links = (Link *) malloc (sizeof(Link) * links_size);
		Link* link = NULL;
		for(size_t i=0; i < links_size; i++) {
			link = new Link();
			links[i] = link->setTask(linksArray[i]);
		}
	}

	/**********************************************************/
	/********              Pod   variables                    ********/
	/*********************************************************/
	const rapidjson::Value &containersArray = task["containers"];
	unsigned int containers_size = containersArray.Size();
	Container **containers = NULL;
	if(containers_size!=0) {
		std::map<unsigned int, Pod*> temp_pods;
		Pod* pod = NULL;
		containers=(Container**) malloc (sizeof(container*)*containers_size);
		Container *c=NULL;
		for(size_t i=0; i < containers_size; i++) {
			c = new Container();
			//CRIA UM CONTAINER E COLOCA OS VALORES DENTRO DELE, APOS ISSO ADICIONA ELE DENTRO DO ARRAY DE CONTAINERS =).

			//EPC_MIN
			c->setEpcMin(containersArray[i]["epc_min"].GetDouble());
			this->epc_min+=c->getEpcMin();

			//EPC_MAX
			c->setEpcMax(containersArray[i]["epc_max"].GetDouble());
			this->epc_max+=c->getEpcMax();

			//RAM_MIN
			c->setRamMin(containersArray[i]["ram_min"].GetDouble());
			this->ram_min+=c->getRamMin();

			//RAM_MAX
			c->setRamMax(containersArray[i]["ram_max"].GetDouble());
			this->ram_max+=c->getRamMax();

			//VCPU_MIN
			c->setVcpuMin(containersArray[i]["vcpu_min"].GetDouble());
			this->vcpu_min+=c->getVcpuMin();

			//VCPU_MAX
			c->setVcpuMax(containersArray[i]["vcpu_max"].GetDouble());
			this->vcpu_max+=c->getVcpuMax();

			//POD
			int pod_index = containersArray[i]["pod"].GetInt();

			// The pod isn't already created, create a new pod and then set the container into it.
			if(temp_pods.find(pod_index) == temp_pods.end()) {
				pod = new Pod(pod_index);
				temp_pods[pod_index] = pod;
			}

			//NAME
			c->setName(containersArray[i]["name"].GetInt());

			temp_pods[pod_index]->addContainer(c);
			containers[i]=c;
		}

		this->pods_size = temp_pods.size();
		//Create the pods array
		this->pods = (Pod**) malloc (sizeof(Pod*)*this->pods_size);

		//Populate the pods array
		for(
			std::map<int,Pod*>::iterator it=temp_pods.begin();
			it!=temp_pods.end();
			it++
			) {
			this->pods[it->first-1] = it->second;
		}
	}

	for(size_t i=0; i<links_size; i++) {
		unsigned int source = links[i]->getSource();
		for(size_t j=0; j< containers_size; j++) {
			if(source = containers[i].getId()) {
				containers[i]->setLink(links[i]);
			}
		}
	}

	free(links);
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

unsigned int Task::getId(){
	return this->id;
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

float Task::getEpcMin(){
	return this->epc_min;
}

float Task::getEpcMax(){
	return this->epc_max;
}

float Task::getRamMin(){
	return this->ram_min;
}

float Task::getRamMax(){
	return this->ram_max;
}

float Task::getVcpuMin(){
	return this->vcpu_min;
}

float Task::getVcpuMax(){
	return this->vcpu_max;
}

std::ostream& operator<<(std::ostream& os, const Task& t)  {
	os<<"Task:{\n";
	os<<"\tDuration: "<<p.duration<<"\n";
	os<<"\tLinks:[";
	for(size_t i=0; i<p.links_size; i++) {
		os<<(*p.links[i])<<",";
	}
	os<<"]\n\tContainers:[\n";
	for(size_t i=0; i<p.containers_size; i++) {
		os<<(*p.containers[i]);
	}
	os<<"\t]\n";
	os<<"\tId: "<< p.id<<"\n";
	os<<"\tSubmission: "<<p.submission<<"\n";
	os<<"\tTotal Resources\n";
	os<<"\t\tepc min: " <<p.epc_min<< "; epc_max: " <<p.epc_max<<"\n";
	os<<"\t\tram min: " <<p.ram_min<< "; ram_max: " <<p.ram_max<<"\n";
	os<<"\t\tvcpu min: "<<p.vcpu_min<<"; vcpu_max: "<<p.vcpu_max<<"\n";
	os<<"}\n";
	return os;
}
