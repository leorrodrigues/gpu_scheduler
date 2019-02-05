#include "pod.hpp"

Pod::Pod(){
	containers=NULL;
	links=NULL;
	duration=0;
	id=0;
	submission=0;

	containers_size = 0;
	links_size=0;
	allocated_time=0;
	delay=0;
	fit=0;

	epc_min=0;
	ram_min=0;
	vcpu_min=0;
	epc_max=0;
	ram_max=0;
	vcpu_max=0;

}

Pod::~Pod(){
	free(containers);
	free(links);
	containers=NULL;
	links=NULL;
}

void Pod::setTask(const char* taskMessage){
	rapidjson::Document task;
	task.Parse(taskMessage);

	/**********************************************************/
	/********                   Pod variables         *********/
	/**********************************************************/
	//Duration
	this->duration = task["duration"].GetDouble();
	//Links
	const rapidjson::Value &linksArray = task["links"];
	this->links_size = linksArray.Size();
	if(this->links_size!=0) {
		this->links = (float *) malloc (sizeof(float) * this->links_size);
		for(size_t i=0; i < this->links_size; i++) {
			this->links[i] = linksArray[i].GetDouble();
		}
	}else{
		this->links=NULL;
	}

	//ID
	this->id = task["id"].GetInt();

	//SUBMISSION
	this->submission = task["submission"].GetDouble();

	/**********************************************************/
	/********              Container variables         ********/
	/**********************************************************/
	Container *c=NULL;
	const rapidjson::Value &containersArray = task["containers"];
	this->containers_size = containersArray.Size();
	if(this->containers_size!=0) {
		this->containers = (Container**) malloc (sizeof(Container*)*this->containers_size);
		for(size_t i=0; i < this->containers_size; i++) {
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
			c->setPod(containersArray[i]["pod"].GetInt());

			//NAME
			c->setName(containersArray[i]["name"].GetInt());

			this->containers[i]=c;
		}
	}else{
		this->containers=0;
	}

}

void Pod::addDelay(){
	this->delay++;
}

void Pod::addDelay(unsigned int delay){
	this->delay+=delay;
}

void Pod::setFit(unsigned int fit){
	this->fit=fit;
}

void Pod::setAllocatedTime(unsigned int allocatedTime){
	this->allocated_time = allocatedTime;
}

void Pod::setSubmission(unsigned int submission){
	this->submission = submission;
}

Container** Pod::getContainers(){
	return this->containers;
}

float* Pod::getLinks(){
	return this->links;
}

unsigned int Pod::getDuration(){
	return this->duration;
}

unsigned int Pod::getId(){
	return this->id;
}

unsigned int Pod::getSubmission(){
	return this->submission;
}

unsigned int Pod::getContainersSize(){
	return this->containers_size;
}

unsigned int Pod::getLinksSize(){
	return this->links_size;
}

unsigned int Pod::getAllocatedTime(){
	return this->allocated_time;
}

unsigned int Pod::getDelay(){
	return this->delay;
}

unsigned int Pod::getFit(){
	return this->fit;
}

float Pod::getEpcMin(){
	return this->epc_min;
}

float Pod::getEpcMax(){
	return this->epc_max;
}

float Pod::getRamMin(){
	return this->ram_min;
}

float Pod::getRamMax(){
	return this->ram_max;
}

float Pod::getVcpuMin(){
	return this->vcpu_min;
}

float Pod::getVcpuMax(){
	return this->vcpu_max;
}

std::ostream& operator<<(std::ostream& os, const Pod& p)  {
	os<<"Pod:{\n";
	os<<"\tDuration: "<<p.duration<<"\n";
	os<<"\tLinks:[";
	for(size_t i=0; i<p.links_size; i++) {
		os<<p.links[i]<<",";
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
