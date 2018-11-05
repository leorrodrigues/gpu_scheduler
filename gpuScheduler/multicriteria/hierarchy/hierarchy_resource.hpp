#ifndef  _HIERARCHY_RESOURCE_NOT_INCLUDED_
#define _HIERARCHY_RESOURCE_NOT_INCLUDED_

#include <cstring>
#include <cstdlib>

class H_Resource {
private:

char** names;
float* data;
int data_size;

public:

H_Resource();
~H_Resource();

void clear();

void addResource(char* name, float value);

float getResource(int index);
float getResource(char* name);

char* getResourceName(int index);

int getDataSize();
};

#endif
