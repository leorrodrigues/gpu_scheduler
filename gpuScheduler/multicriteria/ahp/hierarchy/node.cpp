#include "node.hpp"
#include "edge.hpp"

Node::Node(){
	this->resources = NULL;
	this->name=NULL;
	this->edges=NULL;
	this->matrix=NULL;
	this->normalized_matrix=NULL;
	this->pml=NULL;
	this->pg=NULL;
	this->leaf=true;
	this->active=true;
	this->size=0;
	this->type= node_t::ALTERNATIVE;
}

Node::~Node(){
	free(this->name);
	delete(this->resources);

	int i;
	if(this->matrix!=NULL)
		free( this->matrix );
	if(this->normalized_matrix!=NULL)
		free( this->normalized_matrix );
	if(this->pml!=NULL)
		free( this->pml );
	if(this->pg!=NULL) {
		free( this->pg );
	}
	this->name=NULL;
	this->resources=NULL;
	this->matrix=NULL;
	this->normalized_matrix=NULL;
	this->pml=NULL;
	this->pg=NULL;

	if(this->edges!=NULL)
		for(i=0; i<this->size; i++)
			delete(this->edges[i]);
	free(this->edges);
	this->edges = NULL;
}

void Node::setResource(H_Resource resource) {
	if ( this->resources == NULL)
		this->resources = new H_Resource();

	int i=0;
	for(i=0; i<resource.getDataSize(); i++) {
		// try to insert the resource, if exists, update their value, but if dont exists, insert in the array.
		this->resources->addResource( resource.getResourceName(i), resource.getResource(i) );
	}
}

void Node::setResource(H_Resource* resource) {
	if ( this->resources == NULL)
		this->resources = new H_Resource();

	int i=0;
	for(i=0; i<resource->getDataSize(); i++) {
		// try to insert the resource, if exists, update their value, but if dont exists, insert in the array.
		this->resources->addResource( resource->getResourceName(i), resource->getResource(i) );
	}
}

void Node::setResource(char* name, float value){
	this->resources->addResource( name, value);
}

void Node::setName(const char* name){
	this->name = (char*) malloc (strlen(name)+1);
	// Set the name for the specific node
	strcpy( this->name, name);
}

void Node::setEdges(Edge** edges){
	int i;
	if( this->edges!=NULL) {
		delete(this->edges);
		this->edges = NULL;
	}

	this->edges = (Edge**) malloc ( sizeof(Edge*)*this->size);

	for(i=0; i<this->size; i++) {
		this->edges[i] = edges[i];
	}
}

void Node::addEdge(Edge* edge){
	this->edges = (Edge**) realloc (this->edges, sizeof(Edge*)*(this->size+1));

	this->edges[this->size] = edge;

	if (this->type==node_t::CRITERIA) {
		this->leaf=false;
	}

	this->size++;
}

void Node::setMatrix(float* matrix){
	this->matrix=matrix;
}

void Node::setNormalizedMatrix(float* nMatrix){
	this->normalized_matrix=nMatrix;
}

void Node::setPml(float* pml){
	this->pml=pml;
}

void Node::setPg(float* pg){
	this->pg=pg;
}

void Node::setTypeFocus(){
	this->type=node_t::FOCUS;
	this->leaf= false;
}

void Node::setTypeCriteria(){
	this->type=node_t::CRITERIA;
}

void Node::setTypeAlternative(){
	this->type=node_t::ALTERNATIVE;
	this->leaf = false;
}

void Node::setLeaf(bool leaf){
	this->leaf = leaf;
}

void Node::setActive(bool active){
	this->active =  active;
}

void Node::setSize(int size){
	this->size = size;
}

H_Resource* Node::getResource(){
	return this->resources;
}

char* Node::getName(){
	return this->name;
}

int Node::getSize(){
	return this->size;
}

Edge** Node::getEdges(){
	return this->edges;
}

float* Node::getMatrix(){
	return this->matrix;
}

float* Node::getNormalizedMatrix(){
	return this->normalized_matrix;
}

float* Node::getPml(){
	return this->pml;
}

float* Node::getPg(){
	return this->pg;
}

bool Node::getLeaf(){
	return this->leaf;
}

bool Node::getActive(){
	return this->active;
}

node_t Node::getType(){
	return this->type;
}

void Node::clearEdges(){
	if( this->edges!=NULL) {
		int i;
		for(i=0; i<this->size; i++) {
			delete(this->edges[i]);
		}
		delete(this->edges);
		this->edges = NULL;
		this->size = 0;
	}
}
