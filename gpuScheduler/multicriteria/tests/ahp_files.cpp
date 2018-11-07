#include "../../thirdparty/catch.hpp"
#include "../ahp.hpp"

#include "../../datacenter/host.hpp"
#include "../../datacenter/resource.hpp"

#include <cstring>
#include <string>
#include <unistd.h>

#include <random>

Resource resource;
std::vector<Host*> hosts;

std::string strLower(std::string s) {
	std::transform(s.begin(), s.end(), s.begin(),[](unsigned char c){
		return std::tolower(c);
	});
	return s;
}

void addResource(std::string name, std::string type){
	if (type == "int") {
		//Check if the type is int.
		//Create new entry in the int map.
		resource.mInt[name] = 0;
		resource.mIntSize++;
	} else if (type == "double" || type == "float") {
		//Check if the type is float or float, the variable will be in the same map.
		//Create new entry in the WeightType map.
		resource.mWeight[name] = 0;
		resource.mWeightSize++;
	} else if (type == "string" || type == "char*" || type == "char[]" || type == "char") {
		//Check if the type is string or other derivative.
		//Create new entry in the std::string map.
		resource.mString[name] = "";
		resource.mStringSize++;
	} else if (type == "bool" || type == "boolean") {
		//Check if the type is bool or boolean.
		//Create the new entry in the bool map.
		resource.mBool[name] = false;
		resource.mBoolSize++;
	} else {
		//If the type is unknow the program exit.
		std::cout << "Builder -> Unrecognizable type\nExiting...\n";
		exit(0);
	}
}

Host* addHost() {
	//Call the host constructor (i.e., new host).
	Host* host = new Host(resource);
	//Add the host pointer in the hierarchy (i.e., the hosts vector).
	hosts.push_back(host);
	return host;
}

void generateContentSchema() {
	std::string names;
	std::string text = "{\"$schema\":\"http://json-schema.org/draft-04/schema#\",\"definitions\":{\"topology\": {\"type\": \"object\",\"minProperties\": 1,\"additionalProperties\": false,\"properties\": {\"type\": {\"type\": \"string\"},\"size\": {\"type\": \"number\"},\"level\": {\"type\": \"number\"}},\"required\": [\"type\",\"size\"]},     \"host\": {\"type\": \"array\",\"minItems\": 1,\"items\":{\"properties\": {";
	for (auto it : resource.mInt) {
		text += "\"" + it.first + "\":{\"type\":\"number\"},";
		names += "\"" + it.first + "\",";
	}
	for (auto it : resource.mWeight) {
		text += "\"" + it.first + "\":{\"type\":\"number\"},";
		names += "\"" + it.first + "\",";
	}
	for (auto it : resource.mBool) {
		text += "\"" + it.first + "\":{\"type\":\"boolean\"},";
		names += "\"" + it.first + "\",";
	}
	for (auto it : resource.mString) {
		text += "\"" + it.first + "\":{\"type\":\"string\"},";
		names += "\"" + it.first + "\",";
	}
	names.pop_back();
	text.pop_back();
	text += "},\"additionalProperties\": false,\"required\": [" + names +        "]}}},\"type\": \"object\",\"minProperties\": 1,\"additionalProperties\": false,\"properties\": {\"topology\": {\"$ref\": \"#/definitions/topology\"}, \"hosts\": {\"$ref\": \"#/definitions/host\"}},\"required\": [\"topology\",\"hosts\"]}";
	JSON::writeJson("datacenter/json/hostsSchema.json", text);
}

void parserResources(JSON::jsonGenericType* dataResource) {
	std::string variableName, variableType;
	for (auto &arrayData : dataResource->value.GetArray()) {
		variableName = variableType = "";
		for (auto &objectData : arrayData.GetObject()) {
			if (strcmp(objectData.name.GetString(), "name") == 0) {
				variableName = objectData.value.GetString();
			} else if (strcmp(objectData.name.GetString(), "variableType") == 0) {
				variableType = strLower(objectData.value.GetString());
			} else {
				std::cout << "Error in reading resources\nExiting...\n";
				exit(0);
			}
		}
		addResource(variableName, variableType);
	}
}

void parserHosts(JSON::jsonGenericType* dataHost) {
	for (auto &arrayHost : dataHost->value.GetArray()) {
		auto host = addHost();
		for (auto &alt : arrayHost.GetObject()) {
			std::string name(alt.name.GetString());
			if (alt.value.IsNumber()) {
				if (host->getResource()->mInt.count(name) > 0) {
					host->setResource(name, alt.value.GetInt());
				} else {
					host->setResource(name, alt.value.GetFloat());
				}
			} else if (alt.value.IsBool()) {
				host->setResource(name, alt.value.GetBool());
			} else {
				host->setResource(name, strLower(std::string(alt.value.GetString())));
			}
		}
	}
}

void parserDOM(JSON::jsonGenericDocument* data) {
	for (auto &m : data->GetObject()) { // query through all objects in data.
		if (strcmp(m.name.GetString(), "resources") == 0) {
			parserResources(&m);
		} else if (strcmp(m.name.GetString(), "hosts") == 0) {
			parserHosts(&m);
		} else if (strcmp(m.name.GetString(),"topology")==0) {
			continue;
		}
	}
}

void parser(
	const char* hostsDataPath,
	const char* resourceDataPath,
	const char* hostsSchemaPath,
	const char* resourceSchemaPath
	){
	//Parser the resources
	rapidjson::SchemaDocument resourcesSchema =
		JSON::generateSchema(resourceSchemaPath);
	rapidjson::Document resourcesData =
		JSON::generateDocument(resourceDataPath);
	rapidjson::SchemaValidator resourcesValidator(resourcesSchema);
	if (!resourcesData.Accept(resourcesValidator))
		JSON::jsonError(&resourcesValidator);
	parserDOM(&resourcesData);
	generateContentSchema();
	//Parser the hosts
	rapidjson::SchemaDocument hostsSchema =
		JSON::generateSchema(hostsSchemaPath);
	rapidjson::Document hostsData =
		JSON::generateDocument(hostsDataPath);
	rapidjson::SchemaValidator hostsValidator(hostsSchema);
	if (!hostsData.Accept(hostsValidator))
		JSON::jsonError(&hostsValidator);
	parserDOM(&hostsData);
}

int main(){

	AHP *ahp = new AHP();
	ahp->setHierarchy();

	parser(
		"../../datacenter/json/fat_tree/4.json",
		"../../datacenter/json/resourcesData.json",
		"../../datacenter/json/hostsSchema.json",
		"../../datacenter/json/resourcesSchema.json"
		);

	ahp->run(&hosts[0], hosts.size());

	delete(ahp);
}
