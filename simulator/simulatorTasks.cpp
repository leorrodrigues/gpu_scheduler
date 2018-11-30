#include "comunicator.hpp"
#include "reader.hpp"

#include "thirdparty/clara.hpp"

typedef struct {
	int type;
	int size;
	std::string path;
} options_t;

void setup(int argc, char** argv, options_t* options){
	int size=0;
	int type=0;
	// dont show the help by default. Use `-h or `--help` to enable it.
	bool showHelp = false;
	auto cli = clara::detail::Help(showHelp)
	           | clara::detail::Opt( size, "size" )["-s"]["--size"]("What is the size of your entry? [ (default) 0]")
	           | clara::detail::Opt( type, "Data Type") ["-d"] ["--data_type"] ("What is the type of your data? [ (Default) 0 for data center | 1 for container]");
	auto result = cli.parse( clara::detail::Args( argc, argv ) );
	if( !result ) {
		std::cerr << "Error in command line: " << result.errorMessage() <<std::endl;
		exit(1);
	}
	if ( showHelp ) {
		std::cout << cli << std::endl;
		exit(0);
	}
	if( size <0 ) {
		std::cerr << "Invalid entered data size\n";
		exit(0);
	}else{
		options->size=size;
	}
	if (type<0 && type>2) {
		std::cerr << "Invalid data type\n";
		exit(0);
	}else{
		options->type=type;
	}
	std::string path="json/";
	if(type==0) {
		path+="datacenter/data-"+std::to_string(size)+".json";
	}else if(type==1) {
		path+="container/data-"+std::to_string(size)+".json";
	}else if(type==2) {
		path+="datacenter/google-"+std::to_string(size)+".json";
	}
	options->path=path;
}


int main(int argc,char **argv){
	options_t* options = new options_t;
	setup(argc, argv, options);
	Reader* reader = new Reader();
	Comunicator *conn= new Comunicator( "localhost", 5672, "test_scheduler");
	// std::cout<<"Creating connection\n";
	conn->setupConnection();
	// std::cout<<"Connected to RabbitMq\n";
	reader->openDocument(options->path.c_str());
	std::string message;
	int i=0;
	while((message=reader->getNextTask())!="eof") {
		//std::cout<<a<<"\n-----\n";
		conn->sendMessage(message);
		i++;
	}
	std::cout<<"Total of sent messages "<<i<<"\n";
	conn->closeConnection();
	delete(options);
	delete(reader);
	return 0;
}
