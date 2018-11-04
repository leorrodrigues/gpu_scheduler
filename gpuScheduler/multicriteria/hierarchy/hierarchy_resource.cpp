#include "hierarchy_resource.hpp"

H_Resource::H_Resource(){
	this->names=NULL;
	this->data=NULL;
	this->data_size=0;
}

H_Resource::~H_Resource(){
	int i=0;
	for(i; i<this->data_size; i++) {
		free(this->names[i]);
	}
	free(this->names);
	free(this->data);
	this->names=NULL;
	this->data=NULL;
}

void H_Resource::addResource(char* name, float value){
	int index;
	// search for the element name, if the name dont exist in the array, insert it
	for(index=0; index<this->data_size; index++) {
		if( strcmp( this->names[index], name ) == 0 ) {
			break;
		}
	}
	if(index>=this->data_size) {  // go to the end of the data array and didn't found the value name. So, creates the new data entry and inset a value on it.
		this->names = (char**)realloc(this->names, sizeof(char*)*this->data_size+1);

		this->names[data_size] = (char*) realloc (this->names[this->data_size],  strlen(name));

		this->data = (float*) realloc(this->data,sizeof(float)*this->data_size+1);

		strcpy(this->names[data_size],name);

		this->data_size++;
	} else { // found the data name and update it
		index = this->data_size;
	}

	this->data[index]=value;
}

float H_Resource::getResource(int index){
	return this->data[index];
}

float H_Resource::getResource(char* name){
	int i=0;
	for(i=0; i<this->data_size; i++) {
		if((this->names[i], name ) == 0) {
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
