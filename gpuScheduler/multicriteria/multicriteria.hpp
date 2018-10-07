#ifndef _MULTICRITERIA_NOT_INCLUDED_
#define _MULTICRITERIA_NOT_INCLUDED_

#include <vector>
#include <map>

#include "../datacenter/host.hpp"
#include "../json.hpp"

typedef std::string VariablesType;
typedef float WeightType;

class Multicriteria {
public:
virtual std::map<std::string,int> getResult()=0;
//virtual void run() =0;
virtual void run(std::vector<Host*> host={}) = 0;
virtual void setAlternatives(std::vector<Host*>)=0;

};

#endif
