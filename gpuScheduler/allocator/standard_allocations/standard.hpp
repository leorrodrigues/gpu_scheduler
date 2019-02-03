#ifndef _STANDARD_NOT_INCLUDED_
#define _STANDARD_NOT_INCLUDED_

#include <vector>
#include <map>

#include "../datacenter/host.hpp"
#include "../json.hpp"

class Standard {
public:

virtual ~Standard()=0;
virtual unsigned int run(Host** host={}, int size=0)=0;
};

inline Standard::~Standard(){
}
#endif
