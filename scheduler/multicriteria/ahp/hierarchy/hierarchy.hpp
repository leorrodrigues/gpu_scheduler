/**
   \file Hierarchy.hpp
    Developed by Leonardo Rosa Rodrigues<br>
    August, 2017.  <br>
 */

#ifndef _HIERARCHY_NOT_INCLUDED_
#define _HIERARCHY_NOT_INCLUDED_

#include <algorithm>
#include <iostream>
#include <cstring>
#include <cstdlib>

//#include "../datacenter/host.hpp"

#include "hierarchy_resource.hpp"
#include "node.hpp"
#include "edge.hpp"

/**
   \class Hierarchy
   \brief Data structure that defines the AHP hierarchy. It is composed by four inner classes and one structure.

   The Hierarchy class is a template class to represent the hierarchy used in the AHP method. This class has two template arguments, the char* and the float*.

   The char* represents the template class argument to define the identifier of each node, usually this argument is std::string and int. If you want to use char* all the variables comparison  need to be changed to strcmp() or need to be created a operator function to equals ('=').

   The float* represents the template class argument to define the weights in the hierarchy, usually in the method the weights are represented by a floating point number (float or double).

   Each node in the hierarchy is one of three classes used to represent a node.
   (I)  Focus        : used to represent the Main Objective in the AHP Method;
   (II) Criteria      : used to represent the criteria node, the criterias has two types, normal criteria and leaf criteria, the difference between them are only the connection (edges) that they can have;
        (I)  Normal criteria:  The edges can be between the Focus node and this node or this node and another normal criteira;
        (II) Leaf Criteria    : The edges can be between normal criteira and this node or with this node and alternative.
   (III)Alternative:  used to represent each alternative in the hierarchy, the alternatives has to be connected only with the leafs criterias.
 */

class Hierarchy {
public:
/**
        \struct Resource
        \brief Resource structure used to manage the variables listed in the resources.json. It's composed by four maps to permit the AHP hierarchy uses any number of variables and types.
                The variables types that were valid is:
                ---------------------|-------------------------------
                Type                 | Representation in Json
                --------------------|--------------------------------
                int                    | int
                (!)float*  | double
                (!)float*  | float
                bool                 | bool
                bool                 | boolean
                std::string        | string
                std::string        | char*
                std::string        | char[]
                                (!) The float* is a template parameter.
 */

/*Constructors*/
Hierarchy();

/*Destructor */
~Hierarchy();

//Check if the hierarchy is empty
bool checkEmpty();

/*Hierarchy creator functions*/
Node* addFocus(const char*);
Node *addCriteria(const char*);
void addEdge(const char*, const char*, float* weight = NULL, int size=0);
void addEdge(Node*, Node*, float* weight=NULL, int size=0);
void addResource(const char*);
void addResource(char*);
Node *addAlternative();
Node *addAlternative(Node*);
void addEdgeCriteriasAlternatives();

/*Printing Status Function*/
void listFocus();
void listCriteria();
void listResources();

/*Finding Functions*/
bool findCriteria(Node *);
Node *findCriteria(const char*);
Node *findAlternative(const char*);

/*Clear Functions*/
void clearCriteriasEdges();
void clearAlternatives();
void clearResource();

/*Getters*/
Node *getFocus();
int getNodesSize();
int getCriteriasSize();
int getAlternativesSize();
H_Resource *getResource();
Node** getCriterias();
Node** getAlternatives();

private:
Node** criterias; //< Criteria pointer array to store all the criterias
size_t criterias_size;

Node**  alternatives;        ///< Alternative pointer vector to store all the alternatives.
size_t alternatives_size;

Node* objective;        ///< The main objective pointer.

H_Resource resource;        ///< The default resources of the hierarchy, all the alternatives uses this default resource variable to initiate their resource variable.

void addEdgeObjective(Node*, float* weight = NULL, int size = 0);
void addEdgeCriteria(Node*, Node*, float* weight = NULL, int size = 0);
void addEdgeAlternative(Node*, Node*, float* weight = NULL, int size = 0);

};
#endif
