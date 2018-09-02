#include "reader.hpp"

Reader::Reader(){
}

void Reader::parser(const char* taskDataPath, const char* taskSchemaPath){
	rapidjson::SchemaDocument taskSchema =
		JSON::generateSchema(taskSchemaPath);
	rapidjson::Document taskData =
		JSON::generateDocument(taskDataPath);
	rapidjson::SchemaValidator taskValidator(taskSchema);
	if (!taskData.Accept(taskValidator))
		JSON::jsonError(&taskValidator);
	parserDOM(&taskData);
}

void Reader::parserDOM(JSON::jsonGenericDocument* document){
	for(auto &m : document->GetObject()) {
		if(strcmp(m.name.GetString(),"tasks")==0) {
			this->parserTasks(&m);
		}
		else{
			std::cout<<"Unrecognizable Object\n";
			exit(1);
		}
	}
}

void Reader::parserTasks(JSON::jsonGenericType* tasksObject){
	std::string taskStr;
	bool hasValue;
	for(auto &taskObject : tasksObject->value.GetArray()) {
		taskStr="{";
		for(auto &variable: taskObject.GetObject()) {
			if(variable.value.IsNumber()) {
				taskStr+="\""+std::string(variable.name.GetString())+"\":"+std::to_string(variable.value.GetDouble())+",";
			}else if(variable.value.IsArray()) {
				taskStr+="\""+std::string(variable.name.GetString())+"\":[";
				for(auto &containerArray : variable.value.GetArray()) {
					taskStr+="{";
					hasValue=false;
					for(auto &containerObj : containerArray.GetObject()) {
						taskStr+="\""+std::string(containerObj.name.GetString())+"\":"+ std::to_string(containerObj.value.GetDouble())+",";
						hasValue=true;
					}
					if(hasValue) taskStr.pop_back();
					taskStr+="}";
				}
				taskStr+="],";
			}
		}
		taskStr.pop_back();
		taskStr+="}";
		std::cout<<taskStr<<"\n\n\n";
		exit(0);
	}
}
