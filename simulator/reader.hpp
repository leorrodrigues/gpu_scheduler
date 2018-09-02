#ifndef _READER_NOT_INCLUDED_
#define _READER_NOT_INCLUDED_

#include "json.hpp"

class Reader {
private:

public:
Reader();

void parser(const char* taskDataPath="json/requestDataDefault.json",const char* taskSchemaPath="json/requestSchema.json");
void parserDOM(JSON::jsonGenericDocument*);
void parserTasks(JSON::jsonGenericType*);
};

#endif
