#include <iostream>

#include "../../../datacenter/host.hpp"
#include "../../multicriteria.hpp"
#include "../../../json.hpp"
#include "../topsis.cuh"

#include <assert.h>
#include <vector>
#include <string>
#include <map>

int main(){
	TOPSIS *topsis = new TOPSIS();
	float price[]={250,200,300,275,225};
	float storage[]={16,16,32,32,16};
	float camera[]={12,8,16,8,16};
	float looks[]={5,3,4,4,2};
	std::vector<Host*> hosts;
	for(int i=0; i<5; i++) {
		std::map<std::string,float> resource;
		resource["price"]=price[i];
		resource["storage"]=storage[i];
		resource["camera"]=camera[i];
		resource["looks"]=looks[i];
		Host *m = new Host(resource);
		hosts.push_back(m);
	}
	topsis->run(&hosts[0], 5);
	unsigned int result_size;
	unsigned int *result = topsis->getResult(result_size);

	unsigned int answer[] = {2,3,4,0,1};
	// printf("Resultado\n");
	for(int i=0; i<result_size; i++) {
		assert(answer[i]==result[i]);
		// printf("%d - ", result[i]);
	}
	// printf("\n");
	printf("All tests passed!\n");
	delete(topsis);
	return 0;
}
