#include "container.hpp"

Container::Container(){
	this->duration=0;
	this->links=NULL;
	this->containerResources = new container_resources_t;
	this->containerResources->name=0;
	this->containerResources->pod=0;
	this->containerResources->epc_min=0;
	this->containerResources->epc_max=0;
	this->containerResources->ram_min=0;
	this->containerResources->ram_max=0;
	this->containerResources->vcpu_min=0;
	this->containerResources->vcpu_max=0;
	this->id=0;
	this->submission=0;
}

void Container::setTask(const char* taskMessage){
	std::string::size_type sz;
	std::string taskStr (taskMessage);
	std::cout<<"CONTAINER:\n"<<taskStr<<"----\n";
	std::regex durationRegex ("duration\":(\\s*)(.*),");
	std::regex linkRegex ("links\":(\\s*)[(.*)],");
	std::regex epc_minRegex ("epc_min\":(\\s*)(.*),");
	std::regex epc_maxRegex ("epc_max\":(\\s*)(.*)\n");
	std::regex ram_minRegex ("ram_min\":(\\s*)(.*),");
	std::regex ram_maxRegex ("ram_max\":(\\s*)(.*),");
	std::regex vcpu_minRegex ("vcpu_min\":(\\s*)(.*),");
	std::regex vcpu_maxRegex ("vcpu_max\":(\\s*)(.*),");
	std::regex podRegex ("pod\":(\\s*)(.*),");
	std::regex nameRegex ("name\":(\\s*)(.*),");
	std::regex idRegex ("id\":(\\s*)(.*),");
	std::regex submissionRegex ("submission\":(\\s*)(.*)\n");
	std::smatch sm;
	//Duration
	std::regex_search (taskStr, sm,durationRegex);
	this->duration = std::stod(sm[2].str(),&sz);

	//Links
	std::regex_search (taskStr, sm,linkRegex);
	//TODO links constuctor

	//EPC_MIN
	std::regex_search (taskStr, sm, epc_minRegex);
	std::cout<<sm[2].str()<<"\n";
	this->containerResources->epc_min = std::stod(sm[2].str(),&sz);

	//EPC_MAX
	std::regex_search (taskStr, sm, epc_maxRegex);
	std::cout<<sm[2].str()<<"\n";
	this->containerResources->epc_max = std::stod(sm[2].str(),&sz);

	//RAM_MIN
	std::regex_search (taskStr, sm, ram_minRegex);
	this->containerResources->ram_min = std::stod(sm[2].str(),&sz);

	//RAM_MAX
	std::regex_search (taskStr, sm, ram_maxRegex);
	this->containerResources->ram_max = std::stod(sm[2].str(),&sz);

	//VCPU_MIN
	std::regex_search (taskStr, sm, vcpu_minRegex);
	this->containerResources->vcpu_min = std::stod(sm[2].str(),&sz);

	//VCPU_MAX
	std::regex_search (taskStr, sm, podRegex);
	this->containerResources->pod = std::stoi(sm[2].str(),&sz);

	//NAME
	std::regex_search (taskStr, sm, nameRegex);
	this->containerResources->name = std::stoi(sm[2].str(),&sz);

	//ID
	std::regex_search (taskStr, sm, idRegex);
	this->id = std::stod(sm[2].str(),&sz);

	//SUBMISSION
	std::regex_search (taskStr, sm, submissionRegex);
	this->submission = std::stod(sm[2].str(),&sz);
}

Container::container_resources_t* Container::getResource(){
	return this->containerResources;
}

double Container::getDuration(){
	return this->duration;
}

int Container::getId(){
	return this->id;
}

double Container::getSubmission(){
	return this->submission;
}
