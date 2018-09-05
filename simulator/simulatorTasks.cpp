#include "comunicator.hpp"
#include "reader.hpp"

int main(int argc,char **argv){
	Reader *reader= new Reader();
	Comunicator *conn= new Comunicator("localhost",5672,"test_scheduler");
	std::cout<<"Creating connection\n";
	conn->setupConnection();
	std::cout<<"Connected to RabbitMq\n";
	reader->openDocument();
	for(int i=0; i<10000; i++) {
		auto a= reader->getNextTask();
		std::cout<<a<<"\n-----\n";
		conn->sendMessage(a);
	}
	conn->closeConnection();
	return 0;
}
