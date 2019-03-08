#include "hierarchy_resource.hpp"

H_Resource::H_Resource(){
	this->names=NULL;
	this->data=NULL;
	this->data_size=0;
}

H_Resource::~H_Resource(){
	int i=0;
	for(; i<this->data_size; i++) {
		free(this->names[i]);
	}
	free(this->names);
	free(this->data);
	this->names=NULL;
	this->data=NULL;
	this->data_size=0;
}

void H_Resource::clear(){
	int i=0;
	for(; i<this->data_size; i++) {
		free(this->names[i]);
	}
	free(this->names);
	free(this->data);
	this->names=NULL;
	this->data=NULL;
	this->data=0;
	this->data_size=0;
}

void H_Resource::addResource(char* name, float value){
	int index;
	// search for the element name, if the name dont exist in the array, insert it
	for(index=0; index<this->data_size; index++) {
		if( strcmp( this->names[index], name ) == 0 ) {
			this->data[index]=value;
			return;
		}
	}

	this->names = (char**)realloc(this->names, sizeof(char*)*(this->data_size+1));

	this->names[this->data_size] = NULL;

	this->names[this->data_size] = (char*) malloc (strlen(name)+1);

	this->data = (float*) realloc(this->data,sizeof(float)*(this->data_size+1));

	strcpy(this->names[data_size], name);

	this->data[data_size]=value;

	this->data_size++;
}

float H_Resource::getResource(int index){
	if(this->data_size <= index) {
		printf("H_RESOURCE Error invalid data acess, data_size %d , index %d\n", this->data_size, index);
		exit(0);
	}
	return this->data[index];
}

float H_Resource::getResource(char* name){
	int i=0;
	for(i=0; i<this->data_size; i++) {
		if(strcmp(this->names[i], name ) == 0) {
			return this->data[i];
		}
	}
	// Simulate Error
	return -1;
}

char* H_Resource::getResourceName(int index){
	return this->names[index];
}

int H_Resource::getDataSize(){
	return this->data_size;
}
