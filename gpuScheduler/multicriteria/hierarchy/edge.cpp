#include "edge.hpp"

/**
    \brief The Edge constructor.
    \param The Node pointer.
    \param The Float array.
    \param The Int Size.
 */
Edge::Edge(Node* node, float* weights, int size){
	this->node = node;
	this->setWeights(weights,size);
}

/**
    \brief Edge destructor.
 */
Edge::~Edge(){
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

/**
    \brief Set function to set the weights vector.
 */
void Edge::setWeights(float* weights, int size){
	free(this->weight);
	this->weight=NULL;
	int i;
	this->weight = (float*) realloc (this->weight, sizeof(float)* size);
	for ( i=0; i<size; i++) {
		this->weight[i] = weights[i];
	}
}

/**
    \brief Set function to set the weights in reverse way.
 */
void Edge::setBackWeights(float* weights, int size){
	free(this->weight);
	this->weight=NULL;
	int i;
	this->weight = (float*) realloc (this->weight, sizeof(float)* size);
	for ( i=size; i>0; i--) {
		this->weight[i] = weights[size-i];
	}
}
