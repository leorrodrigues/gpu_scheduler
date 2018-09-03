#include "json.hpp"

namespace JSON {

typedef rapidjson::GenericMember<rapidjson::UTF8<char>, rapidjson::MemoryPoolAllocator<rapidjson::CrtAllocator> > jsonGenericType;

rapidjson::Document* generateDocument(const char *path){
	FILE* fp = fopen(path, "r");
	fseek(fp, 0, SEEK_END);
	size_t filesize = (size_t)ftell(fp);
	fseek(fp, 0, SEEK_SET);
	char* buffer = (char*)malloc(filesize + 1);
	size_t readLength = fread(buffer, 1, filesize, fp);
	buffer[readLength] = '\0';
	fclose(fp);
	rapidjson::Document* sd = new rapidjson::Document;
	if(sd->Parse(buffer).HasParseError()) {
		free(buffer);
		std::cout<<"Erro in parsing the json";
	}
	return sd;
}

rapidjson::SchemaDocument generateSchema(const char* path){
	rapidjson::Document* sd=generateDocument(path);
	rapidjson::SchemaDocument schema(*sd);
	return schema;
}

void jsonError(rapidjson::SchemaValidator* validator){
// Input JSON is invalid according to the schema
// Output diagnostic information
	rapidjson::StringBuffer sb;
	validator->GetInvalidSchemaPointer().StringifyUriFragment(sb);
	printf("Invalid schema: %s\n", sb.GetString());
	printf("Invalid keyword: %s\n", validator->GetInvalidSchemaKeyword());
	sb.Clear();
	validator->GetInvalidDocumentPointer().StringifyUriFragment(sb);
	printf("Invalid document: %s\n", sb.GetString());
	exit(0);
}

void writeJson(const char* path,std::string text){
	std::ofstream jsonFile (path, std::ios::out | std::ios::trunc);
	if (jsonFile.is_open()) {
		jsonFile << text;
		jsonFile.close();
	}
}
}
