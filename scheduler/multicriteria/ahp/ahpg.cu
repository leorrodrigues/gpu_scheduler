#include "ahpg.cuh"

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		SPDLOG_ERROR("CUDA Runtime Error: {}", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

AHPG::AHPG() {
	IR[3] = 0.5245;
	IR[4] = 0.8815;
	IR[5] = 1.1086;
	IR[6] = 1.1279;
	IR[7] = 1.3417;
	IR[8] = 1.4056;
	IR[9] = 1.4499;
	IR[10] = 1.4854;
	IR[11] = 1.5141;
	IR[12] = 1.5365;
	IR[13] = 1.5551;
	IR[14] = 1.5713;
	IR[15] = 1.5838;
	char cwd[1024];
	char* result;
	result = getcwd(cwd, sizeof(cwd));
	if(result == NULL) {
		SPDLOG_ERROR("The directory path is not correct");
	}
	char* sub_path = (strstr(cwd, "ahp"));
	if(sub_path!=NULL) {
		int position = sub_path - cwd;
		strncpy(this->path, cwd, position);
		this->path[position-1] = '\0';
	}else{
		strcpy(this->path, cwd);
		strcat(this->path,"/");
	}
}

AHPG::~AHPG(){
	delete(hierarchy);
}

void AHPG::run(Host** alternatives={}, int size=0){

}

unsigned int* AHPG::getResult(unsigned int&){
	unsigned int* result = (unsigned int*) malloc (sizeof(unsigned int));

	unsigned int i;

	float* values = this->hierarchy->getFocus()->getPg();

	std::priority_queue<std::pair<float, int> > alternativesPair;

	// spdlog::debug("PG\n");
	for (i = 0; i < (unsigned int) this->hierarchy->getAlternativesSize(); i++) {

		// if( (unsigned int)alternativesPair.size()<20)
		// spdlog::debug("%f#%u$ ",values[i],i);
		alternativesPair.push(std::make_pair(values[i], i));
	}
	// spdlog::debug("\n");


	Node** alternatives = this->hierarchy->getAlternatives();

	i=0;
	while(!alternativesPair.empty()) {
		result = (unsigned int*) realloc (result, sizeof(unsigned int)*(i+1));
		result[i] = atoi(alternatives[alternativesPair.top().second]->getName());
		alternativesPair.pop();
		i++;
	}
	size = i;
	return result;
}

void setAlternatives(Host** host,int size){

}

void hierarchyParserG(const rapidjson::Value &hierarchyData){

}
