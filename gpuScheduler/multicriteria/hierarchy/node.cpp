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
	free(name);
	delete(resources);

	int i;
	for(i=0; i<this->size; i++) {
		free( matrix[i] );
		free( normalized_matrix[i] );
	}
	free( matrix );
	free( normalized_matrix );
	free( pml );
	free( pg );
	this->name=NULL;
	this->resources=NULL;
	this->matrix=NULL;
	this->normalized_matrix=NULL;
	this->pml=NULL;
	this->pg=NULL;
}

void Node::setResource(H_Resource resource) {
	if ( this->resources = NULL)
		this->resources = new H_Resource();

	int i=0;
	for(i=0; i<resource.getDataSize(); i++) {
		// try to insert the resource, if exists, update their value, but if dont exists, insert in the array.
		this->resources->addResource( resource.getResourceName(i), resource.getResource(i) );
	}
}

void Node::setName(const char* name){
	// Set the name for the specific node
	strcpy( this->name, name);
}

void Node::setEdges(Edge** edges){
	int i;
	if( this->edges!=NULL) {
		delete(this->edges);
		this->edges = NULL;
	}

	this->edges = (Edge**) realloc (this->edges, sizeof(Edge*)*this->size);

	for(i=0; i<this->size; i++) {
		this->edges[i] = edges[i];
	}
}

void Node::addEdge(Edge* edge){
	this->edges = (Edge**) realloc (this->edges, sizeof(Edge*)*this->size+1);

	this->edges[this->size] = edge;

	this->size++;
	if (this->type==node_t::CRITERIA) {
		this->leaf=false;
	}
}

void Node::setMatrix(float** matrix){
	int i,j;
	if( this->matrix != NULL) {
		for(i=0; i<this->size; i++) {
			free(this->matrix[i]);
			this->matrix[i]=NULL;
		}
		free(this->matrix);
		this->matrix = NULL;
	}
	this->matrix = (float**) realloc (this->matrix, sizeof(float*) * this->size);

	for(i=0; i<size; i++) {
		this->matrix[i]= (float*) realloc (this->matrix[i], sizeof(float) * this->size);
		for( j=0; j<size; j++) {
			this->matrix[i][j]= matrix[i][j];
		}
	}
}

void Node::setNormalizedMatrix(float** nMatrix){
	int i,j;
	if( this->normalized_matrix != NULL ) {
		for(i=0; i<this->size; i++) {
			free(this->normalized_matrix[i]);
			this->normalized_matrix[i]=NULL;
		}
		free(this->normalized_matrix);
		this->normalized_matrix = NULL;
	}
	this->normalized_matrix = (float**) realloc ( this->normalized_matrix, sizeof(float*)  * this->size );

	for( i=0; i<this->size; i++) {
		this->normalized_matrix[i] = (float*) realloc (this->normalized_matrix[i], sizeof(float) * this->size);
		for( j=0; j<this->size; j++) {
			this->normalized_matrix[i][j] = nMatrix[i][j];
		}
	}
}

void Node::setPml(float* pml){
	if( this->pml != NULL) {
		free( this->pml);
		this->pml = NULL;
	}
	int i;
	this->pml = (float*) realloc ( this->pml, sizeof(float) * this->size );
	for(i=0; i<this->size; i++) {
		this->pml[i] = pml[i];
	}
}

void Node::setPg(float* pg){
	if( this->pg != NULL ) {
		free(this->pg);
		this->pg=NULL;
	}
	int i;
	this->pg =  (float*) realloc ( this->pg, sizeof(float) * this->size );
	for(i=0; i<this->size; i++) {
		this->pg[i] = pg[i];
	}
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

float** Node::getMatrix(){
	return this->matrix;
}

float** Node::getNormalizedMatrix(){
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

void Node::clearEdges(){
	if( this->edges!=NULL) {
		delete(this->edges);
		this->edges = NULL;
	}
}
