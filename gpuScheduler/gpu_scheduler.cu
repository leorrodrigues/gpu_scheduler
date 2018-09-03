#include <iostream>
#include <chrono>
#include <ctime>
#include <ratio>

#include "builder.cuh"

int main(int argc, char **argv){
	Comunicator *conn = new Comunicator("localhost",5672,"test_scheduler");
	conn->setupConnection();
	for(int i=0; i<10; i++)
		std::cout<<conn->getMessage()<<"\n";

	//std::cout<<"Starting...\n";
	std::chrono::steady_clock::time_point pI, pF, mI,mF, cI,cF,crI,crF, mrI,mrF,aI,aF;
	Builder *builder= new Builder();
	//std::cout<<"Parsing...\n";
	if(argc==2) {
		//std::cout<<argv[1];
		pI=std::chrono::steady_clock::now();
		builder->parser(argv[1]);
		pF=std::chrono::steady_clock::now();
	}else{
		builder->parser("datacenter/json/hostsDataDefault.json");
	}
	//std::cout<<"Setting AHP\n";
	builder->setAHP();
	//std::cout<<"Setting MCL\n";
	//builder->setMCL();
	//builder->getTopology()->listTopology();
	//builder->setBcube(2,2);
	//builder->printTopologyType();
	//builder->setDcell(2,2);
	//builder->printTopologyType();
	if(builder->getHosts().size()>0) {
		aI=std::chrono::steady_clock::now();
		//std::cout<<"Running MCL\n";
		//cI=std::chrono::steady_clock::now();

		//	builder->runClustering(builder->getHosts());

		//cF=std::chrono::steady_clock::now();
		//std::cout<<"Get MCL results\n";
		//crI=std::chrono::steady_clock::now();

		//builder->getClusteringResult();

		//crF=std::chrono::steady_clock::now();
		//std::cout<<"Running AHP\n";
		mI=std::chrono::steady_clock::now();

		//builder->runMulticriteria( builder->getClusterHosts() );
		builder->runMulticriteria(builder->getHosts());
		mF=std::chrono::steady_clock::now();
		//std::cout<<"Getting AHP results\n";
		//builder->listCluster();
		mrI=std::chrono::steady_clock::now();

		auto results=builder->getMulcriteriaResult();

		aF=std::chrono::steady_clock::now();
		mrF=std::chrono::steady_clock::now();
		//for(auto it: results) {
		//	std::cout<<it.first<<" "<<it.second<<"\n";
		//}
	}else{
		std::cout<<"Alternatives with 0 size\n";
	}
	//std::cout<<"End\n";
	std::chrono::duration<double> parser_span = std::chrono::duration_cast<std::chrono::duration<double> >(pF - pI);
	std::chrono::duration<double> multicriteria_span = std::chrono::duration_cast<std::chrono::duration<double> >(mF - mI);
	//std::chrono::duration<double> cluster_span = std::chrono::duration_cast<std::chrono::duration<double> >(cF - cI);
	//std::chrono::duration<double> cluster_get_span = std::chrono::duration_cast<std::chrono::duration<double> >(crF - crI);
	std::chrono::duration<double> multicriteria_get_span = std::chrono::duration_cast<std::chrono::duration<double> >(mrF - mrI);
	std::chrono::duration<double> all_span = std::chrono::duration_cast<std::chrono::duration<double> >(aF - aI);

	std::cout << "Parser: " << parser_span.count() << " seconds.\n";
	//std::cout << "Cluster: " << cluster_span.count() << " seconds.\n";
	std::cout << "Multicriteria: " << multicriteria_span.count() << " seconds.\n";
	//std::cout << "Cluster Resource: " << cluster_get_span.count() << " seconds.\n";
	std::cout << "Multicriteria Resource: " << multicriteria_get_span.count() << " seconds.\n";
	std::cout << "All: " << all_span.count() << " seconds.\n";
	return 0;
}
