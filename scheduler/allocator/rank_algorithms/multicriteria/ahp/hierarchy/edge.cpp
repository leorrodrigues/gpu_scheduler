#include "edge.hpp"

/**
    \brief The Edge constructor.
    \param The Node pointer.
    \param The Float array.
    \param The Int Size.
 */
Edge::Edge(Node* node, float* weights, int size){
	this->size = 0;
	this->node = node;
	this->weight=NULL;
	this->setWeights(weights,size);
}

/**
    \brief Edge destructor.
 */
Edge::~Edge(){
	this->size=0;
	this->node=NULL;
	free(this->weight);
}

/**
    \brief Get function to return the weight.
    \return Weight vector.
 */
float* Edge::getWeights(){
	return this->weight;
}

/**
    \brief Get function to return the node.
    \return Node Pointer.
 */
Node* Edge::getNode(){
	return this->node;
}

int Edge::getSize(){
	return this->size;
}

/**
    \brief Set function to set the weights vector.
 */
void Edge::setWeights(float* weights, int size){
	if(size==0) return;
	if(this->weight != NULL && this->size>0) {
		free(this->weight);
		this->weight=NULL;
	}
	int i;
	this->weight = (float*) malloc ( sizeof(float)* size);
	for ( i=0; i<size; i++) {
		this->weight[i] = weights[i];
	}
	this->size=size;
}

/**
    \brief Set function to set the weights in reverse way.
 */
void Edge::setBackWeights(float* weights, int size){
	if(size==0) return;
	if(this->weight != NULL && this->size>0) {
		free(this->weight);
		this->weight=NULL;
	}
	int i;
	this->weight = (float*) malloc ( sizeof(float)* size);
	for ( i=size; i>0; i--) {
		this->weight[i] = weights[size-i];
	}
}
