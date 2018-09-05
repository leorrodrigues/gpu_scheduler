#ifndef _READER_NOT_INCLUDED_
#define _READER_NOT_INCLUDED_

#include "json.hpp"

class Reader {
private:
JSON::jsonGenericDocument *doc;
int index;
public:
Reader(){
	doc = NULL;
	index = 0;
}

void openDocument(const char* taskDataPath="json/requestDataDefault.json",const char* taskSchemaPath="json/requestSchema.json"){
	rapidjson::SchemaDocument taskSchema =
		JSON::generateSchema(taskSchemaPath);
	JSON::jsonGenericDocument* taskData =                                       JSON::generateDocument(taskDataPath);
	rapidjson::SchemaValidator taskValidator(taskSchema);
	if (!taskData->Accept(taskValidator))
		JSON::jsonError(&taskValidator);
	this->doc = taskData;

}

std::string getNextTask(){
	const rapidjson::Value& taskArray = (*this->doc)["tasks"];
	if(taskArray.Size()<=(unsigned int)this->index) return "eof";

	rapidjson::StringBuffer buffer;

	rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);

	const rapidjson::Value& task = taskArray[this->index];

	task.Accept(writer);

	this->index++;

	return std::string(buffer.GetString());
}

};

#endif
