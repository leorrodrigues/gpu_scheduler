#ifndef _NODE_NOT_INCLUDED_
#define _NODE_NOT_INCLUDED_

#include <cstdlib>
#include <cstring>

#include "hierarchy_resource.hpp"
// #include "edge.hpp"

class Edge;
/**
    \class Node
        \brief Node  class to represent all the tree types of node that the hierarchy has , the focus, criteria and alternative node.
 */
typedef enum {
	FOCUS, CRITERIA, ALTERNATIVE
} node_t;

class Node {
public:
Node();
~Node();

void setResource(H_Resource);
void setResource(char*,float);
void setName(const char*);
void setEdges(Edge**);
void addEdge(Edge*);
void setLeaf(bool);
void setActive(bool);
void setSize(int);
void setMatrix(float**);
void setNormalizedMatrix(float**);
void setPml(float*);
void setPg(float*);
void setTypeFocus();
void setTypeCriteria();
void setTypeAlternative();

H_Resource* getResource();
char* getName();
int getSize();
Edge** getEdges();
bool getLeaf();
bool getActive();
node_t getType();
float** getMatrix();
float** getNormalizedMatrix();
float* getPml();
float* getPg();

void clearEdges();

private:
node_t type;

H_Resource* resources;  ///< Resource variable to store the variables.

char* name;  ///< Variable to store the main objective name or id.

Edge** edges;  ///< Vector of edges pointers to represent all the links between any two nodes.

float** matrix; ///< Matrix used to represent the weights matrix used in the acquisition AHP step.
float** normalized_matrix;  ///< Normalized Matrix used to represent the values that were obtained by the normalization of the matrix values.
float* pml; ///< PML variable is used to store all the local avarage priority of the criterias. (*) Different to focus class, the criterias don't have PG variable.
float* pg; ///< PG variable is used to store the Global priority of the alternatives, represents the final alternative's ranking.

int size; ///< Size variable is used to represent the size of all array elements.

bool leaf; ///< Leaf variable is used to assist in the management of the final hierarchical nodes. If this node is true the node will be included in the criterias and sheets vector, otherwise it's included only in the criterias vector.

bool active; ///< Active variable is used to assist in the AHP methods. If this variable is true, the respective node is used in the hierarchy, but if this node it's inactive, this node and all theis childs aren't  used in the AHP methods.
};
#endif
