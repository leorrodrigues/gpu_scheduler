#ifndef _JSON_NOT_INCLUDED_
#define _JSON_NOT_INCLUDED_

#include <fstream>
#include <iostream>

#include "thirdparty/rapidjson/schema.h"
#include "thirdparty/rapidjson/pointer.h"
#include "thirdparty/rapidjson/prettywriter.h"

#include "thirdparty/rapidjson/filereadstream.h"

typedef rapidjson::GenericMember<rapidjson::UTF8<char>, rapidjson::MemoryPoolAllocator<rapidjson::CrtAllocator> > genericValue;

namespace JSON {

typedef rapidjson::GenericMember<rapidjson::UTF8<char>, rapidjson::MemoryPoolAllocator<rapidjson::CrtAllocator> > jsonGenericType;
typedef rapidjson::GenericDocument<rapidjson::UTF8<char>, rapidjson::MemoryPoolAllocator<rapidjson::CrtAllocator>, rapidjson::CrtAllocator> jsonGenericDocument;
rapidjson::Document generateDocument(const char *path);
rapidjson::Document* generateDocumentP(const char *path);
rapidjson::SchemaDocument generateSchema(const char* path);
void jsonError(rapidjson::SchemaValidator* validator);
void writeJson(const char* path,std::string text);
};
#endif
