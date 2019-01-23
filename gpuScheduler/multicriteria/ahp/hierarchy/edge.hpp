#ifndef _EDGE_NOT_DEFINED_
#define _EDGE_NOT_DEFINED_
/**
    \class Edge
        \brief Edge class to represent the connection of two nodes in the hierarchy.
 */

 #include <cstdlib>

#include "node.hpp"

class Node;

class Edge {

public:
Edge(Node*, float*,int);
~Edge();

float* getWeights();
Node *getNode();
int getSize();

void setWeights(float*,int);
void setBackWeights(float*,int);

private:
Node* node;         ///< Node Pointer to represent the link to an criteria.

float* weight;         ///< Array of Weights to represent the weights in the edge having the size of all the alternatives/criterias in the same hierarchy level. Ex: if the hierarchy has 4 alternatives, the vector contains [x1,x2,x3,x4] weights, where the weight 'i' in the alternative 'i' has their value 1.
int size;
};

#endif
