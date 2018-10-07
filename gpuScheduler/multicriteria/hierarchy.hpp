/**
   \file Hierarchy.hpp
    Developed by Leonardo Rosa Rodrigues<br>
    August, 2017.  <br>
 */

#ifndef _HIERARCHY_NOT_INCLUDED_
#define _HIERARCHY_NOT_INCLUDED_

#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "../datacenter/host.hpp"

/**
   \class Hierarchy
   \brief Data structure that defines the AHP hierarchy. It is composed by four inner classes and one structure.

   The Hierarchy class is a template class to represent the hierarchy used in the AHP method. This class has two template arguments, the VariablesType and the WeightType.

   The VariablesType represents the template class argument to define the identifier of each node, usually this argument is std::string and int. If you want to use char* all the variables comparison  need to be changed to strcmp() or need to be created a operator function to equals ('=').

   The WeightType represents the template class argument to define the weights in the hierarchy, usually in the method the weights are represented by a floating point number (float or double).

   Each node in the hierarchy is one of three classes used to represent a node.
   (I)  Focus        : used to represent the Main Objective in the AHP Method;
   (II) Criteria      : used to represent the criteria node, the criterias has two types, normal criteria and leaf criteria, the difference between them are only the connection (edges) that they can have;
        (I)  Normal criteria:  The edges can be between the Focus node and this node or this node and another normal criteira;
        (II) Leaf Criteria    : The edges can be between normal criteira and this node or with this node and alternative.
   (III)Alternative:  used to represent each alternative in the hierarchy, the alternatives has to be connected only with the leafs criterias.
 */

template <class VariablesType, class WeightType> class Hierarchy {
public:
/**
        \struct Resource
        \brief Resource structure used to manage the variables listed in the resources.json. It's composed by four maps to permit the AHP hierarchy uses any number of variables and types.
                The variables types that were valid is:
                ---------------------|-------------------------------
                Type                 | Representation in Json
                --------------------|--------------------------------
                int                    | int
                (!)WeightType  | double
                (!)WeightType  | float
                bool                 | bool
                bool                 | boolean
                std::string        | string
                std::string        | char*
                std::string        | char[]
                                (!) The WeightType is a template parameter.
 */
typedef struct {
	std::map<std::string, int> mInt;         ///< map to represent all the int variables.
	std::map<std::string, WeightType> mWeight;         ///< map to represent all the float and double variables.
	std::map<std::string, std::string> mString;         ///< map to represent all the strings variables.
	std::map<std::string, bool> mBool;        ///< map to represent all bool variables.
	int mIntSize, mWeightSize, mStringSize, mBoolSize;        ///< variables to represent theis respective map size, used to reduce the function overloads call (removing the call of .size() in the map).
} Resource;

//Inner classes declaration.
class Edge;
class Alternative;
class Focus;
class Criteria;

//Typedef used to reduce declaration variables size.
typedef typename std::vector<Alternative *>::iterator alternativeIt;
typedef typename std::vector<Criteria *>::iterator criteriaIt;
typedef typename std::vector<Edge *>::iterator edgeIt;

/**
    \class Edge
        \brief Edge class to represent the connection of two nodes in the hierarchy.

        One edge can be setting up between (I) Focus and Criteria nodes; (II) Criteria and Criteria nodes; (III) Criteria and Leaf Criteria nodes; or (IV) Leaf Criteria and Alternative nodes.
 */
class Edge {
//friend classes declaration.
friend class Hierarchy;
friend class Criteria;
friend class Focus;

public:
Edge(Criteria *, std::vector<WeightType>);
Edge(Alternative *, std::vector<WeightType>);
~Edge();

std::vector<WeightType> getWeights();
Criteria *getCriteria();
Alternative *getAlternative();

void setWeights(std::vector<WeightType>);

private:
Criteria *cCrit;         ///< Criteria Pointer to represent the link to an criteria.
Alternative *cAlt;         ///< Alternative Pointer to represent the link to an alternative.
std::vector<WeightType> weight;         ///< Vector of Weights to represent the weights in the edge having the size of all the alternatives/criterias in the same hierarchy level. Ex: if the hierarchy has 4 alternatives, the vector contains [x1,x2,x3,x4] weights, where the weight 'i' in the alternative 'i' has their value 1.
};

/**
    \class Alternative
        \brief Alternative class to represent all the alternatives in the hierarchy.  The alternatives uses the resource structure to represent any type and amount of variables.
 */
class Alternative {
friend class Hierarchy;

public:
Alternative(Hierarchy<VariablesType, WeightType>::Resource);
Alternative(Host* a);
~Alternative();

void setResource(std::string, int);
void setResource(std::string, bool);
void setResource(std::string, std::string);
void setResource(std::string, WeightType);

Hierarchy<VariablesType, WeightType>::Resource *getResource();
std::string getName();

private:
Resource resources;                 ///< Resource variable to store the variables.
};

/**
    \class Focus
        \brief Focus class to represent the main objective in the hierarchy.
 */
class Focus {
friend class Hierarchy;

public:
Focus(const VariablesType);
~Focus();

std::vector<Edge *> getEdges();
WeightType **getMatrix();
WeightType **getNormalizedMatrix();
WeightType *getPml();
WeightType *getPg();
VariablesType getName();

void setMatrix(WeightType **);
void setNormalizedMatrix(WeightType **);
void setPml(WeightType *);
void setPg(WeightType *);

int edgesCount();

private:
VariablesType name;         ///< Variable to store the main objective name or id.
std::vector<Edge *> edges;        ///< Vector of edges pointers to represent all the links between the focus and the criterias.
WeightType **matrix;        ///< Matrix used to represent the weights matrix used in the acquisition AHP step.
WeightType **normalizedMatrix;        ///< Normalized Matrix used to represent the values that were obtained by the normalization of the matrix values.
WeightType *pml;        ///< PML variable is used to store all the local avarage priority of the criterias.
WeightType *pg;        ///< PG variable is used to store the Global priority of the alternatives, represents the final alternative's ranking.
};

/**
    \class Criteria
        \brief Criteira class to represent all the intermediate nodes in the hierarchy.
 */
class Criteria {
friend class Hierarchy;

public:
Criteria(const VariablesType name);
~Criteria();

std::vector<Edge *> getEdges();
WeightType **getMatrix();
WeightType **getNormalizedMatrix();
WeightType *getPml();
VariablesType getName();
bool getLeaf();
bool getActive();
Criteria *getParent();
Focus *getOParent();

void setLeaf(bool);
void setMatrix(WeightType **);
void setNormalizedMatrix(WeightType **);
void setPml(WeightType *);

int edgesCount();

private:
VariablesType name;         ///< Variable to store the node name or id.
std::vector<Edge *> edges;        ///< Vector of edges to store all the childs in the hierarch (i.e., all the links between this node and other nodes).
Criteria *parent;        ///< Criteria pointer to the parent.
Focus *oParent;        ///< Focus pointer to the parent. The construction of the criteria parent and focus parent is needed to facilitate the methods constructions (bottom-up methods).
WeightType **matrix;        ///< Matrix used to represent the weights matrix used in the acquisition AHP step.
WeightType **normalizedMatrix;        ///< Normalized Matrix used to represent the values that were obtained by the normalization of the matrix values.
WeightType *pml;        ///< PML variable is used to store all the local avarage priority of the criterias. (*) Different to focus class, the criterias don't have PG variable.
bool leaf;        ///< Leaf variable is used to assist in the management of the final hierarchical nodes. If this node is true the node will be included in the criterias and sheets vector, otherwise it's included only in the criterias vector.
bool active;        ///< Active variable is used to assist in the AHP methods. If this variable is true, the respective node is used in the hierarchy, but if this node it's inactive, this node and all theis childs aren't  used in the AHP methods.
};
/*Constructors*/
Hierarchy();

/*Destructor */
~Hierarchy();

/*Hierarchy creator functions*/
Focus *addFocus(const VariablesType);
Criteria *addCriteria(VariablesType);
void addEdge(const VariablesType, const VariablesType,
             std::vector<WeightType> weight = {});
void addEdge(Focus *, Criteria *, std::vector<WeightType> weight = {});
void addEdge(Criteria *, Criteria *, std::vector<WeightType> weight = {});
void addEdge(Criteria *, Alternative *, std::vector<WeightType> weight = {});
void addResource(std::string, std::string);
Alternative *addAlternative();
Alternative *addAlternative(Alternative*);
void addEdgeSheetsAlternatives();
void addSheets(Criteria *);

/*Printing Status Function*/
void listFocus();
void listCriteria();
void listResources();
void show();

/*Finding Functions*/
bool findCriteria(const Criteria *);
Criteria *findCriteria(const VariablesType);
bool findSheets(const Criteria *);
Alternative *findAlternative(const VariablesType);

/*Clear Functions*/
void clearSheetsEdges();
void clearAlternatives();

/*Updating Member Functions*/
void updateSheetsEdges();
void updateCriteriaActive(VariablesType, bool);
void updateCriteriaActive(Criteria *, bool);

/*Getters*/
Focus *getFocus();
int getSheetsCount();
int getAlternativesCount();
Resource *getResource();
std::vector<Criteria *> getCriterias();
std::vector<Criteria *> getSheets();
std::vector<Alternative *> getAlternatives();

private:
void listAll(Criteria *, std::string str, std::vector<WeightType> weight);        ///< Helper function to list all informations about the hierarchy.
std::vector<Criteria *> criterias;        ///< Criterias pointer vector to store all the criterias of the hierarchy.
std::vector<Criteria *> sheets;        ///< Criteria pointer vector to store all the leaf nodes in the hierarchy.
std::vector<Alternative *> alternatives;        ///< Alternative pointer vector to store all the alternatives.
Focus *objective;        ///< The main objective pointer.
Resource resource;        ///< The default resources of the hierarchy, all the alternatives uses this default resource variable to initiate their resource variable.
};

/***************************************************************
**********************HIERARCHY***************************
***************************************************************/

/**
        \brief Hierarchy constructior, setting all the variables to default (empty) values.
 */
template <class VariablesType, class WeightType>
Hierarchy<VariablesType, WeightType>::Hierarchy() {
	this->objective = NULL;
	resource.mIntSize = 0;
	resource.mWeightSize = 0;
	resource.mStringSize = 0;
	resource.mBoolSize = 0;
}

/**
        \brief Hierarchy destructor.
 */
template <class VariablesType, class WeightType>
Hierarchy<VariablesType, WeightType>::~Hierarchy() {
}

/**
    \brief Call the Focus constructor and the add it's pointer in the hierarchy.
        \param variable name/id
        \return a Focus pointer or NULL.
 */
template <class VariablesType, class WeightType>
typename Hierarchy<VariablesType, WeightType>::Focus *
Hierarchy<VariablesType, WeightType>::addFocus(const VariablesType name) {
	//The main Objective can't be overwrite
	if (this->objective != NULL)
		return NULL;
	//Call the Focus constructor (i.e., create a new Focus).
	Focus *ob = new Focus(name);
	//Set the memory address of the objective in the hierarchy.
	this->objective = ob;
	//return the Focus pointer.
	return ob;
}

/**
    \brief Call the criteria constructor and them add it in the hierarchy.
    The function will search in the criterias trying to find one criteria with same Name/Id passed as parameter, if any criteria match with the parameter, the function'll return NULL, otherwise the function will call the Criteria constructor and then add this memory address in the criterias vector.
        \param The variable name/id.
        \return The criteria pointer or NULL.
 */
template <class VariablesType, class WeightType>
typename Hierarchy<VariablesType, WeightType>::Criteria *
Hierarchy<VariablesType, WeightType>::addCriteria(VariablesType name) {
	//Iterate though criterias vector.
	for (criteriaIt it = this->criterias.begin(); it != this->criterias.end(); it++) {
		//If found some criteria with same name/id return NULL
		if ((*it)->name == name) {
			return NULL;
		}
	}
	//Call the criteria's constructor (i.e., create a new criteria)
	Criteria *c = new Criteria(name);
	//Add the criteria in hierarchy (criterias vector).
	this->criterias.push_back(c);
	//return the  Criteria pointer.
	return c;
}

/**
    \brief Add an edge between two criterias
    This function will search for the nodes in the criterias vector and them add a edge between them.
    \param parentName: father name.
    \param childName: child name.
    \param w: the edge weight. Defaut value 0.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::addEdge(const VariablesType parentName, VariablesType childName,std::vector<WeightType> w) {
	//Search for some criteria with same name/id.
	Criteria *parent = findCriteria(parentName);
	//Search for some criteria with same name/id.
	Criteria *child = findCriteria(childName);
	//if one of the two nodes not exists in the hierarchy exit this function.
	if (parent == NULL || child == NULL) {
		return;
	}
	//Call the Edge constructor (i.e., create a new edge)
	Edge *edge = new Edge(child, w);
	//Add the edge address in the hierarchy (i.e., edges vector)
	parent->edges.push_back(edge);
	//set the parent address in the child node.
	child->parent = parent;
}

/**
    \brief Add an edge between Focus and Criteria nodes.
    \param objective: Focus pointer.
    \param child: Criteria pointer.
    \param w: the edge weight. Default 0.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::addEdge(Focus *objective, Criteria *child, std::vector<WeightType> w) {
	//Call the edge constructor (i.e., creates a new edge).
	Edge *edge = new Edge(child, w);
	//Add this edge in the Focus edges list.
	objective->edges.push_back(edge);
	//Set the objective as the criterias father.
	child->oParent = objective;
	//Set the criteria parent equals NULL.
	child->parent = NULL;
}

/**
   \brief Add an edge between two Criteria nodes.
   \param parent: Criteria pointer as father.
   \param child: Criteria pointer as child.
   \param w: the edge weight. Default 0.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::addEdge(Criteria *parent, Criteria *child, std::vector<WeightType> w) {
	//Call the edge constructor (i.e., create a new edge).
	Edge *edge = new Edge(child, w);
	//Add this edge in the father edges list.
	parent->edges.push_back(edge);
	//Set the criteria father.
	child->parent = parent;
	//Set the objective father equals NULL.
	child->oParent = NULL;
}

/**
   \brief Add an edge between Criteria and Alternative nodes.
   \param parent: Criteria pointer.
   \param alt: Alternative pointer.
   \param w: the edge weight. Default 0.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::addEdge(Criteria *parent, Alternative *alt, std::vector<WeightType> w) {
	//Call the edge constructor (i.e., create new edge).
	Edge *edge = new Edge(alt, w);
	//Add the edge in the parent edges list.
	parent->edges.push_back(edge);
}

/**
    \brief Add one resource in the hierarchy default resources.
    This function will add new resource in the hierarchy default resources, 4 types in c++ are used (int, float, std::string and bool). Meanwhile, the function will be able to manage 9 types in the json files (int, float, double, string, char*, char[], char, bool and boolean).
    \param name: The resource name.
    \param type: The resource type.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::addResource(std::string name, std::string type) {
	if (type == "int") {
		//Check if the type is int.
		//Create new entry in the int map.
		this->resource.mInt[name] = 0;
		this->resource.mIntSize++;
	} else if (type == "float" || type == "double") {
		//Check if the type is float or double, the variable will be in the same map.
		//Create new entry in the WeightType map.
		this->resource.mWeight[name] = 0;
		this->resource.mWeightSize++;
	} else if (type == "string" || type == "char*" || type == "char[]" || type == "char") {
		//Check if the type is string or other derivative.
		//Create new entry in the std::string map.
		this->resource.mString[name] = "";
		this->resource.mStringSize++;
	} else if (type == "bool" || type == "boolean") {
		//Check if the type is bool or boolean.
		//Create the new entry in the bool map.
		this->resource.mBool[name] = false;
		this->resource.mBoolSize++;
	} else {
		//If the type is unknow the program exit.
		std::cout << "Hierarchy -> Unrecognizable type: "<<type<<"\nExiting...\n";
		exit(0);
	}
}

/**
    \brief Add an alternative pointer in the hierarchy.
    \return An alternative pointer.
 */
template <class VariablesType, class WeightType>
typename Hierarchy<VariablesType, WeightType>::Alternative *
Hierarchy<VariablesType, WeightType>::addAlternative() {
	//Call the alternative constructor (i.e., new alternative).
	Alternative *alternative = new Alternative(this->resource);
	//Add the alternative pointer in the hierarchy (i.e., the alternatives vector).
	this->alternatives.push_back(alternative);
	return alternative;
}

/**
    \brief Add an alternative pointer in the hierarchy.
    \return An alternative pointer.
 */
template <class VariablesType, class WeightType>
typename Hierarchy<VariablesType, WeightType>::Alternative*
Hierarchy<VariablesType, WeightType>::addAlternative(Alternative* alt) {
	//Add the alternative pointer in the hierarchy (i.e., the alternatives vector).
	this->alternatives.push_back(alt);
	return alt;
}

/**
    \brief Add an edge between the hierarchy sheets and the alternatives.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::addEdgeSheetsAlternatives() {
	//Iterate through all the sheets.
	for (criteriaIt it = sheets.begin(); it != sheets.end(); it++) {
		//Iterate through all the alternatives.
		for (alternativeIt ait = alternatives.begin(); ait != alternatives.end();
		     ait++) {
			//Create an edge between them.
			addEdge((*it), (*ait));
		}
	}
}

/**
    \brief Add a new sheet.
    This function will add the existing criteria in the sheets vector if this criteria has the leaf value set as true.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::addSheets(Criteria *c) {
	//If the criteria isn't a sheet.
	if (!findSheets(c)) {
		//Add the criteria in the sheets vector.
		this->sheets.push_back(c);
	}
}

/*Printing status Function*/

/**
    \brief List the main objective name.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::listFocus() {
	//If the objective exists, print their name.
	if (this->objective != NULL) {
		std::cout << "Focus " << this->objective->name << "\n";
	}else{
		std::cout << "Empty Focus\n";
	}
}

/**
   \brief List all the criterias names in the hierarchy.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::listCriteria() {
	//Iterate through the criterias list and print their names.
	for (criteriaIt it = criterias.begin(); it != criterias.end(); it++) {
		std::cout << (*it)->name << "\n";
	}
}

/**
    \brief List all the resources in the default resource of hierarchy.
    The function will check the size of each map and if their size is bigger than 0, the variable name and value will be printed.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::listResources() {
	if (this->resource.mIntSize) {
		std::cout << "Int Resources\n";
		for (auto it : this->resource.mInt) {
			std::cout << "\t" << it.first << " : " << it.second << "\n";
		}
	}
	if (this->resource.mWeightSize) {
		std::cout << "Float/Double Resources\n";
		for (auto it : this->resource.mWeight) {
			std::cout << "\t" << it.first << " : " << it.second << "\n";
		}
	}
	if (this->resource.mStringSize) {
		std::cout << "String Resources\n";
		for (auto it : this->resource.mString) {
			std::cout << "\t" << it.first << " : " << it.second << "\n";
		}
	}
	if (this->resource.mBoolSize) {
		std::cout << "Boolean Resources\n";
		for (auto it : this->resource.mBool) {
			std::cout << "\t" << it.first << " : " << it.second << "\n";
		}
	}
}

/**
    \brief Show all the hierarchy variables and their values.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::show() {
	//Resources are the string to be printed.
	//Add the objective name in the resources.
	std::string resources = this->objective->name + " | ";
	//Iterate through all the edges of the objective.
	for (edgeIt it = this->objective->edges.begin(); it != this->objective->edges.end(); it++) {
		//Call the listAll function with the focus child and the weights.
		listAll((*it)->cCrit, resources, (*it)->weight);
	}
	//Print the alternatives and their resources.
	std::cout << "Alternatives\n";
	for (alternativeIt alt = this->alternatives.begin();
	     alt != this->alternatives.end(); alt++) {
		std::cout << (*alt)->getName() << " # ";
		if ((*alt)->resources.mIntSize) {
			for (auto it : (*alt)->resources.mInt) {
				std::cout << it.first << " : " << it.second << " ; ";
			}
		}
		if ((*alt)->resources.mWeightSize) {
			for (auto it : (*alt)->resources.mWeight) {
				std::cout << it.first << " : " << it.second << " ; ";
			}
		}
		if ((*alt)->resources.mStringSize) {
			for (auto it : (*alt)->resources.mString) {
				std::cout << it.first << " : " << it.second << " ; ";
			}
		}
		if ((*alt)->resources.mBoolSize) {
			for (auto it : (*alt)->resources.mBool) {
				std::cout << it.first << " : " << it.second << " ; ";
			}
		}
		std::cout << "\n";
	}
}

/**
    \brief List all the variables of certain criteria.
    \param c: a criteria pointer to be listed.
    \param str: the string to concatenate the criterias variables values.
    \param weight: a vector of WeightType.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::listAll(Criteria *c, std::string str, std::vector<WeightType> weight) {
	//Concatenate the criteria name.
	str += "( " + c->name + " <-> [";
	//Iterate through the weights vector and concatenate theis values.
	for (WeightType w : weight) {
		str += std::to_string(w) + " ; ";
	}
	str += " ], active= " + std::to_string(c->active) + " ) | ";
	//If the criteria is a leaf, concatenate the remain variables and then return the function.
	if (c->leaf) {
		for (edgeIt it = c->edges.begin(); it != c->edges.end(); it++) {
			str += "( " + (*it)->cAlt->getName() + " <-> [ ";
			for (WeightType w : (*it)->weight)
				str += std::to_string(w) + " ; ";
			str += " ]) | ";
		}
		std::cout << str << "\n";
		return;
	}
	//If the criteria isn't a leaf, iterate through the criterias edges and call recursively the function.
	for (edgeIt it = c->edges.begin(); it != c->edges.end(); it++) {
		listAll((*it)->cCrit, str, (*it)->weight);
	}
}

/*Finding Functions*/

/**
    \brief Search for an specific criteria in the criterias list.
    \param c: Criteria point to be sought.
    \return True if the criteria was found, false otherwise.
 */
template <class VariablesType, class WeightType>
bool Hierarchy<VariablesType, WeightType>::findCriteria(const Criteria *c) {
	//iterate through the criterias in the hierarchy.
	for (criteriaIt it = criterias.begin(); it != criterias.end(); it++)
		if ((*it)->name == c->name)
			return true;
	return false;
}

/**
    \brief Search for an specific criteria thorught its name/id.
    \param name: The name/id to search.
    \return The criteria pointer if exists one criteria with the name parameter, NULL otherwise.
 */
template <class VariablesType, class WeightType>
typename Hierarchy<VariablesType, WeightType>::Criteria *
Hierarchy<VariablesType, WeightType>::findCriteria(const VariablesType name) {
	//Iterate through the criterias list.
	for (criteriaIt it = criterias.begin(); it != criterias.end(); it++) {
		//If the criteria has been found, return it.
		if ((*it)->name == name) {
			return (*it);
		}
	}
	return NULL;
}

/**
    \brief  The function will search the sheets list trying to find the a specific criteria in the list.
    \param c: Criteria pointer.
    \return True if the criteria are in the sheets list, false otherwise.
 */
template <class VariablesType, class WeightType>
bool Hierarchy<VariablesType, WeightType>::findSheets(const Criteria *c) {
	//Iterate through all the sheets list.
	for (criteriaIt it = sheets.begin(); it != sheets.end(); it++)
		if ((*it)->name == c->name)
			return true;
	return false;
}

/**
    \brief The function will search the alternatives list trying to find an alternative with the respective param name/id.
    \param v: The alternative name/id to find.
    \return Alternative pointer if exista an alternative with the respective name/id, NULL otherwise.
 */
template <class VariablesType, class WeightType>
typename Hierarchy<VariablesType, WeightType>::Alternative *
Hierarchy<VariablesType, WeightType>::findAlternative(const VariablesType v) {
	for (alternativeIt it = this->alternatives.begin(); it != this->alternatives.end(); it++) {
		if ((*it)->name == v)
			return (*it);
	}
	return NULL;
}

/*Getting Focus*/

/**
    \brief The function return the main objective pointer.
    \return The focus pointer
 */
template <class VariablesType, class WeightType>
typename Hierarchy<VariablesType, WeightType>::Focus *
Hierarchy<VariablesType, WeightType>::getFocus() {
	return this->objective;
}

/**
    \brief Return the size of the sheets list.
    \return Sheets size.
 */
template <class VariablesType, class WeightType>
int Hierarchy<VariablesType, WeightType>::getSheetsCount() {
	return this->sheets.size();
}

/**
    \brief Return the size of the alternatives list.
    \return Alternatives size.
 */
template <class VariablesType, class WeightType>
int Hierarchy<VariablesType, WeightType>::getAlternativesCount() {
	return this->alternatives.size();
}

/**
    \brief Get function to return the resource pointer.
    \return Return resource pointer.
 */
template <class VariablesType, class WeightType>
typename Hierarchy<VariablesType, WeightType>::Resource *
Hierarchy<VariablesType, WeightType>::getResource() {
	return &(this->resource);
}

/**
    \brief Get function to return the criteria pointer.
    \return Return criteria pointer.
 */
template <class VariablesType, class WeightType>
std::vector<typename Hierarchy<VariablesType, WeightType>::Criteria *>
Hierarchy<VariablesType, WeightType>::getCriterias() {
	return this->criterias;
}

/**
    \brief Get function to return the sheets list.
    \return Return sheets list.
 */
template <class VariablesType, class WeightType>
std::vector<typename Hierarchy<VariablesType, WeightType>::Criteria *>
Hierarchy<VariablesType, WeightType>::getSheets() {
	return this->sheets;
}

/**
    \brief Get function to return the alternatives list.
    \return Return alternatives list.
 */
template <class VariablesType, class WeightType>
std::vector<typename Hierarchy<VariablesType, WeightType>::Alternative *>
Hierarchy<VariablesType, WeightType>::getAlternatives() {
	return this->alternatives;
}

/*Clear Functions*/

/**
    \brief The function will clear all the edges between the sheets and alternatives.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::clearSheetsEdges() {
	//Iterate through all the sheets.
	for (criteriaIt it = this->sheets.begin(); it != this->sheets.end(); it++) {
		(*it)->edges.clear();
	}
}

/**
    \brief The function will clear all the alternatives in the hierarchy.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::clearAlternatives() {
	this->alternatives.clear();
	clearSheetsEdges();
}

/*Update Functions*/

/**
    \brief The function update all the edges between the sheets list and the criterias.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::updateSheetsEdges() {
	clearSheetsEdges();
	addEdgeSheetsAlternatives();
}

/**
    \brief The function update the active variable in a specific criteria.
    \param name of the criteria.
    \param active is the bool value of the active criteria.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::updateCriteriaActive( VariablesType name, bool active) {
	//Search for the criteria by his name
	Criteria *c = findCriteria(name);
	//If the criteria has been found.
	if (c != NULL) {
		//Update the active variable.
		c->active = active;
		//Update the active variable of all the criteria childs.
		for (edgeIt it = c->edges.begin(); it != c->edges.end(); it++) {
			updateCriteriaActive((*it)->cCrit, active);
		}
	}
}

/**
    \brief The function update the active variable in a specific criteria.
    \param The criteria pointer.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::updateCriteriaActive(Criteria *c, bool active) {
	//Check if the pointer isn't null.
	if (c != NULL) {
		//Update the active variable.
		c->active = active;
		//Update the active variable of all the criteria childs.
		for (edgeIt it = c->edges.begin(); it != c->edges.end(); it++) {
			updateCriteriaActive((*it)->cCrit, active);
		}
	}
}

/***************************************************************
**********************EDGE**********************************
***************************************************************/

/**
    \brief The Edge constructor.
    \param The criteria pointer.
    \param The WeightType vector.
 */
template <class VariablesType, class WeightType>
Hierarchy<VariablesType, WeightType>::Edge::Edge(Criteria *c, std::vector<WeightType> w) {
	this->cCrit = c;
	this->cAlt = NULL;
	for (auto it = w.begin(); it != w.end(); it++) {
		this->weight.push_back(*it);
	}
}

/**
    \brief The Edge constructor.
    \param The Alternative pointer.
    \param The WeightType vector.
 */
template <class VariablesType, class WeightType>
Hierarchy<VariablesType, WeightType>::Edge::Edge(Alternative *a, std::vector<WeightType> w) {
	this->cCrit = NULL;
	this->cAlt = a;
	for (auto it = w.begin(); it != w.end(); it++) {
		this->weight.push_back(*it);
	}
}

/**
    \brief Edge destructor.
 */
template <class VariablesType, class WeightType>
Hierarchy<VariablesType, WeightType>::Edge::~Edge() {
}

/**
    \brief Get function to return the weight.
    \return Weight vector.
 */
template <class VariablesType, class WeightType>
std::vector<WeightType>
Hierarchy<VariablesType, WeightType>::Edge::getWeights() {
	return this->weight;
}

/**
    \brief Get function to return the criteria.
    \return Criteria Pointer.
 */
template <class VariablesType, class WeightType>
typename Hierarchy<VariablesType, WeightType>::Criteria *
Hierarchy<VariablesType, WeightType>::Edge::getCriteria() {
	return this->cCrit;
}

/**
    \brief Get function to return the alternative.
    \return Alternative Pointer.
 */
template <class VariablesType, class WeightType>
typename Hierarchy<VariablesType, WeightType>::Alternative *
Hierarchy<VariablesType, WeightType>::Edge::getAlternative() {
	return this->cAlt;
}

/**
    \brief Set function to set the weights vector.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::Edge::setWeights( std::vector<WeightType> w) {
	this->weight.clear();
	copy(w.begin(), w.end(), back_inserter(this->weight));
}

/***************************************************************
**********************ALTERNATIVE*************************
***************************************************************/

/**
    \brief Alternative constructor.
    Set the default variables through the resource variable param.
    \param Resource variable.
 */
template <class VariablesType, class WeightType>
Hierarchy<VariablesType, WeightType>::Alternative::Alternative(
	Hierarchy<VariablesType, WeightType>::Resource resource) {
	resources.mIntSize = resource.mIntSize;
	resources.mWeightSize = resource.mWeightSize;
	resources.mStringSize = resource.mStringSize;
	resources.mBoolSize = resource.mBoolSize;
	resources.mInt = resource.mInt;
	resources.mWeight = resource.mWeight;
	resources.mString = resource.mString;
	resources.mBool = resource.mBool;
}

template <class VariablesType, class WeightType>
Hierarchy<VariablesType, WeightType>::Alternative::Alternative(
	Host* a) {
	resources.mIntSize = a->getResource()->mIntSize;
	resources.mWeightSize = a->getResource()->mWeightSize;
	resources.mStringSize = a->getResource()->mStringSize;
	resources.mBoolSize = a->getResource()->mBoolSize;
	resources.mInt = a->getResource()->mInt;
	resources.mWeight = a->getResource()->mWeight;
	resources.mString = a->getResource()->mString;
	resources.mBool = a->getResource()->mBool;
}

/**
    \brief The alternative's destructor.
 */
template <class VariablesType, class WeightType>
Hierarchy<VariablesType, WeightType>::Alternative::~Alternative() {
}

/**
    \brief Function to set the int resources.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::Alternative::setResource(
	std::string name, int v) {
	this->resources.mInt[name] = v;
}

/**
    \brief Function to set the bool resources.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::Alternative::setResource(
	std::string name, bool v) {
	this->resources.mBool[name] = v;
}

/**
    \brief Function to set the string resources.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::Alternative::setResource(
	std::string name, std::string v) {
	this->resources.mString[name] = v;
}

/**
    \brief Function to set the WeightType resources.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::Alternative::setResource(
	std::string name, WeightType v) {
	this->resources.mWeight[name] = v;
}

/**
    \brief Function to get the resources pointer.
    \return Resources pointer.
 */
template <class VariablesType, class WeightType>
typename Hierarchy<VariablesType, WeightType>::Resource *
Hierarchy<VariablesType, WeightType>::Alternative::getResource() {
	return &(this->resources);
}

/**
    \brief Function to get the alternative name.
    \return Alternative name.
 */
template <class VariablesType, class WeightType>
std::string Hierarchy<VariablesType, WeightType>::Alternative::getName() {
	auto it = this->resources.mString.find("name");
	if (it != this->resources.mString.end()) {
		return it->second;
	}
	return "";
}

/***************************************************************
**********************FOCUS*********************************
***************************************************************/

/**
    \brief The Focus constructor with default values.
    \param The name/id of the Focus.
 */
template <class VariablesType, class WeightType>
Hierarchy<VariablesType, WeightType>::Focus::Focus(const VariablesType name) {
	this->name = name;
	this->matrix = NULL;
	this->normalizedMatrix = NULL;
	this->pg = NULL;
	this->pml = NULL;
}

/**
    \brief Focus destructor.
 */
template <class VariablesType, class WeightType>
Hierarchy<VariablesType, WeightType>::Focus::~Focus() {
}

/*Getters*/

/**
    \brief Functions to get the edges.
    \return Edges vector.
 */
template <class VariablesType, class WeightType>
std::vector<typename Hierarchy<VariablesType, WeightType>::Edge *>
Hierarchy<VariablesType, WeightType>::Focus::getEdges() {
	return this->edges;
}

/**
    \brief Function to get the Matrix.
    \return Matrix of Weights.
 */
template <class VariablesType, class WeightType>
WeightType **Hierarchy<VariablesType, WeightType>::Focus::getMatrix() {
	return this->matrix;
}

/**
    \brief Function to get the Normalized Matrix.
    \return Normalized Matrix of Weights.
 */
template <class VariablesType, class WeightType>
WeightType **
Hierarchy<VariablesType, WeightType>::Focus::getNormalizedMatrix() {
	return this->normalizedMatrix;
}

/**
    \brief Function to get the PML.
    \return PML.
 */
template <class VariablesType, class WeightType>
WeightType *Hierarchy<VariablesType, WeightType>::Focus::getPml() {
	return this->pml;
}

/**
    \brief Function to get the PG.
    \return PG.
 */
template <class VariablesType, class WeightType>
WeightType *Hierarchy<VariablesType, WeightType>::Focus::getPg() {
	return this->pg;
}

/**
    \brief Function to get the Focus name.
    \return Focus name.
 */
template <class VariablesType, class WeightType>
VariablesType Hierarchy<VariablesType, WeightType>::Focus::getName() {
	return this->name;
}

/*Setters*/

/**
    \brief Function to set the matrix in the Focus.
    \param The Weight Matrix.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::Focus::setMatrix(WeightType **m) {
	this->matrix = m;
}

/**
    \brief Function to set the normalized matrix in the Focus.
    \param The Weight normalized Matrix.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::Focus::setNormalizedMatrix(
	WeightType **m) {
	this->normalizedMatrix = m;
}

/**
    \brief Function to set the pml in the Focus.
    \param The PML.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::Focus::setPml(WeightType *pml) {
	this->pml = pml;
}

/**
    \brief Function to set the PG in the Focus.
    \param The PG of alternatives.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::Focus::setPg(WeightType *pg) {
	this->pg = pg;
}

/**
    \brief Function to get the Edges list size.
    \return the edges size.
 */
template <class VariablesType, class WeightType>
int Hierarchy<VariablesType, WeightType>::Focus::edgesCount() {
	return this->edges.size();
}

/***************************************************************
**********************CRITERIA******************************
***************************************************************/

/**
    \brief The Criteria constructor set the variables to default.
    \param The name of the Criteria.
 */
template <class VariablesType, class WeightType>
Hierarchy<VariablesType, WeightType>::Criteria::Criteria(const VariablesType name) {
	this->name = name;
	this->parent = parent;
	this->oParent = NULL;
	this->matrix = NULL;
	this->normalizedMatrix = NULL;
	this->pml = NULL;
	this->leaf = false;
	this->active = true;
}

/**
    \brief Criteria destructor
 */
template <class VariablesType, class WeightType>
Hierarchy<VariablesType, WeightType>::Criteria::~Criteria() {
}

/*Getters*/

/**
    \brief  Function to get the Edges.
    \return Edges vector.
 */
template <class VariablesType, class WeightType>
std::vector<typename Hierarchy<VariablesType, WeightType>::Edge *>
Hierarchy<VariablesType, WeightType>::Criteria::getEdges() {
	return this->edges;
}

/**
    \brief Function to get the matrix.
    \return Weight matrix.
 */
template <class VariablesType, class WeightType>
WeightType **Hierarchy<VariablesType, WeightType>::Criteria::getMatrix() {
	return this->matrix;
}

/**
    \brief Function to get the normalized matrix.
    \return Weight normalized matrix.
 */
template <class VariablesType, class WeightType>
WeightType **
Hierarchy<VariablesType, WeightType>::Criteria::getNormalizedMatrix() {
	return this->normalizedMatrix;
}

/**
    \brief Function to get the PML.
    \return PML.
 */
template <class VariablesType, class WeightType>
WeightType *Hierarchy<VariablesType, WeightType>::Criteria::getPml() {
	return this->pml;
}

/**
    \brief Function to get the criteria name.
    \return criteria name.
 */
template <class VariablesType, class WeightType>
VariablesType Hierarchy<VariablesType, WeightType>::Criteria::getName() {
	return this->name;
}

/**
    \brief Function to get the leaf status.
    \return Leaf variable.
 */
template <class VariablesType, class WeightType>
bool Hierarchy<VariablesType, WeightType>::Criteria::getLeaf() {
	return this->leaf;
}

/**
    \brief  Function to get the Active status.
    \return Active variable.
 */
template <class VariablesType, class WeightType>
bool Hierarchy<VariablesType, WeightType>::Criteria::getActive() {
	return this->active;
}

/**
    \brief Function to get the parent of the criteria.
    \return Criteria pointer.
 */
template <class VariablesType, class WeightType>
typename Hierarchy<VariablesType, WeightType>::Criteria *
Hierarchy<VariablesType, WeightType>::Criteria::getParent() {
	return this->parent;
}

/**
    \brief Function to get the Object parent.
    \return Focus pointer.
 */
template <class VariablesType, class WeightType>
typename Hierarchy<VariablesType, WeightType>::Focus *
Hierarchy<VariablesType, WeightType>::Criteria::getOParent() {
	return this->oParent;
}

/*Setters*/

/**
    \brief Function to set the leaf in the criteria.
    \param The boolean value of leaf status.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::Criteria::setLeaf(bool v) {
	this->leaf = v;
}

/**
    \brief Function to set the matrix in the criteria.
    \param The matrix of weights.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::Criteria::setMatrix(WeightType **m) {
	this->matrix = m;
}

/**
    \brief Function to set the normalized matrix in the criteria.
    \param The normalized matrix of weights.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::Criteria::setNormalizedMatrix(
	WeightType **m) {
	this->normalizedMatrix = m;
}

/**
    \brief Function to set the PML in the criteria.
    \param The PML.
 */
template <class VariablesType, class WeightType>
void Hierarchy<VariablesType, WeightType>::Criteria::setPml(WeightType *pml) {
	this->pml = pml;
}

/**
    \brief Function to get the edges size.
    \param Edges size.
 */
template <class VariablesType, class WeightType>
int Hierarchy<VariablesType, WeightType>::Criteria::edgesCount() {
	return this->edges.size();
}

#endif
