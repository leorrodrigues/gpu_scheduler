#ifndef _MULTICRITERIA_NOT_INCLUDED_
#define _MULTICRITERIA_NOT_INCLUDED_

#include "../rank.hpp"

class Multicriteria : public Rank {
public:

virtual ~Multicriteria() = 0;
virtual unsigned int* getResult(unsigned int& size) = 0;
//virtual void run() =0;
virtual void run(std::vector<Host*> alt, int alt_size, int interval_low, int interval_high) = 0;
virtual void setAlternatives(Host** alternatives, int size, int low, int high) = 0;
virtual void readJson() = 0;
};

inline Multicriteria::~Multicriteria(){
}
#endif
