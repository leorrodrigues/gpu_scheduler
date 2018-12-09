#ifndef _MULTICRITERIA_NOT_INCLUDED_
#define _MULTICRITERIA_NOT_INCLUDED_

#include <vector>
#include <map>

#include "../datacenter/host.hpp"
#include "../json.hpp"

class Multicriteria {
public:

virtual ~Multicriteria()=0;
virtual std::map<int,const char*> getResult()=0;
//virtual void run() =0;
virtual void run(Host** alternatives={}, int size=0) = 0;
virtual void setAlternatives(Host** host, int size)=0;
};

inline Multicriteria::~Multicriteria(){
}
#endif
