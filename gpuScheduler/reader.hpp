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

~Reader(){
	delete(this->doc);
}

void openDocument(const char* taskDataPath, const char* taskSchemaPath = "../simulator/json/containerRequestSchema.json"){
	rapidjson::SchemaDocument taskSchema =
		JSON::generateSchema(taskSchemaPath);
	JSON::jsonGenericDocument* taskData =                                       JSON::generateDocumentP ( taskDataPath );
	rapidjson::SchemaValidator taskValidator(taskSchema);
	if (!taskData->Accept(taskValidator))
		JSON::jsonError(&taskValidator);
	this->doc = taskData;
	taskData = NULL;
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
