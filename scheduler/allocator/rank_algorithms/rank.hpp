#ifndef _RANK_NOT_INCLUDED_
#define _RANK_NOT_INCLUDED_

#include <vector>
#include <map>

#include "../../datacenter/host.hpp"
#include "../../json.hpp"

class Rank {
protected:
unsigned int type=0;
public:
void setType(unsigned int type){
	this->type=type;
}

virtual ~Rank();
virtual unsigned int* getResult(unsigned int& size) = 0;
virtual void run(std::vector<Host*> alt, int alt_size, int interval_low, int interval_high) = 0;
virtual void readJson() = 0;

};

inline Rank::~Rank(){
}
#endif
