#include "pod.hpp"

Pod::Pod(){
	containers=NULL;
	containers_size = 0;
	epc_min=0;
	epc_max=0;
	ram_min=0;
	ram_max=0;
	vcpu_max=0;
	vcpu_min=0;
}

Pod::~Pod(){
	free(containers);
	containers=NULL;
}


void Container::setTask(const char* taskMessage){
	rapidjson::Document task;
	task.parse(taskMessage);

	/**********************************************************/
	/********                   Pod variables                 ******** /
	   /**********************************************************/
	//Duration
	this->duration = task["duration"].GetDouble();
	//Links
	const Value &linksArray = task["links"];
	this->links_size = linksArray.Size();
	if(this->links_size!=0) {
		this->links = (float *) malloc (sizeof(float) * this->links_size);
		for(int i=0; i < this->links_size; i++) {
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
	/********              Container variables             ******** /
	   /**********************************************************/

	const Value &containersArray = task["containers"];
	this->containers_size = containersArray.Size();
	if(this->containers_size!=0) {
		this->containers = (Container**) malloc (sizeof(Container*)*this->containers_size);
		for(int i=0; i < this->containers_size; i++) {
			//CRIA UM CONTAINER E COLOCA OS VALORES DENTRO DELE, APOS ISSO ADICIONA ELE DENTRO DO ARRAY DE CONTAINERS =).

			//EPC_MIN
			this->containerResources->epc_min = containersArray[i] std::stod(sm[2].str(),&sz);

			//EPC_MAX
			this->containerResources->epc_max = std::stod(sm[2].str(),&sz);

			//RAM_MIN
			this->containerResources->ram_min = std::stod(sm[2].str(),&sz);

			//RAM_MAX
			this->containerResources->ram_max = std::stod(sm[2].str(),&sz);

			//VCPU_MIN
			this->containerResources->vcpu_min = std::stod(sm[2].str(),&sz);

			//VCPU_MAX
			this->containerResources->vcpu_max = std::stod(sm[2].str(),&sz);

			//POD
			this->containerResources->pod = std::stoi(sm[2].str(),&sz);

			//NAME
			this->containerResources->name = std::stoi(sm[2].str(),&sz);
		}
	}else{
		this->containers=0;
	}


}

void Pod::addContainer(Container* c){
	Container::container_resources_t* res = c->getResource();
	epc_min  += res->epc_min;
	epc_max += res->epc_max;
	ram_min  += res->ram_min;
	ram_max += res->ram_max;
	epc_min  += res->epc_min;
	epc_max += res->epc_max;
	containers_size++;
	containers = (Container**) realloc (containers, sizeof(Container*)*containers_size);
	containers[containers_size-1] = c;
}
