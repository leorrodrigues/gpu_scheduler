#ifndef  _MAIN_RESOURCE_NOT_DEFINED_
#define _MAIN_RESOURCE_NOT_DEFINED_

#include <string>
#include <map>

#include "../json.hpp"
#include "interval_tree.hpp"

typedef struct main_resource_t {
	std::map<std::string, Interval_Tree::Interval_Tree*> resource;

	explicit main_resource_t(){
		rapidjson::SchemaDocument resourceSchema =
			JSON::generateSchema("main_resources/resourcesSchema.json");
		rapidjson::Document resourceData =
			JSON::generateDocument("main_resources/resourcesData.json");
		rapidjson::SchemaValidator resourceValidator(resourceSchema);
		if (!resourceData.Accept(resourceValidator))
			JSON::jsonError(&resourceValidator);

		const rapidjson::Value &r_array = resourceData["resources"];

		for(size_t i=0; i<r_array.Size(); i++) {
			resource[r_array[i]["name"].GetString()]= new Interval_Tree::Interval_Tree();
		}
	}

} main_resource_t;

#endif
