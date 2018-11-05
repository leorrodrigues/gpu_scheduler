#include "hierarchy.hpp"

/**
        \brief Hierarchy constructior, setting all the variables to default (empty) values.
 */
Hierarchy::Hierarchy() {
	this->nodes=NULL;
	this->nodes_size=0;
	this->sheets=NULL;
	this->sheets_size=0;
	this->alternatives=NULL;
	this->alternatives_size=0;
	this->criterias=NULL;
	this->criterias_size=0;
	this->objective=NULL;
}

/**
        \brief Hierarchy destructor.
 */
Hierarchy::~Hierarchy() {
	int i;
	for(i=0; i< this->nodes_size; i++) {
		delete(this->nodes[i]);
	}
	free(this->nodes);
	for(i=0; i< this->criterias_size; i++) {
		delete(this->criterias[i]);
	}
	free(this->criterias);
	for(i=0; i< this->sheets_size; i++) {
		delete(this->sheets[i]);
	}
	free(this->sheets);
	for(i=0; i< this->alternatives_size; i++) {
		delete(this->alternatives[i]);
	}
	free(this->alternatives);
	delete(this->objective);
	this->nodes=NULL;
	this->criterias=NULL;
	this->sheets=NULL;
	this->alternatives=NULL;
	this->objective=NULL;
}

/**
    \brief Call the Focus constructor and the add it's pointer in the hierarchy.
        \param variable name/id
        \return a Node pointer ( Focus ) or NULL.
 */
Node* Hierarchy::addFocus(const char* name) {
	//The main Objective can't be overwrite
	if (this->objective != NULL)
		return NULL;
	//Call the Focus constructor (i.e., create a new Focus).
	Node *ob = new Node();
	ob->setName(name);
	ob->setTypeFocus();
	//Set the memory address of the objective in the hierarchy.
	this->objective = ob;

	this->nodes = (Node**) realloc (this->nodes, sizeof(Node*)* this->nodes_size+1);

	this->nodes[this->nodes_size] = ob;

	this->nodes_size++;
	//return the Focus pointer.
	return ob;
}

/**
    \brief Call the criteria constructor and them add it in the hierarchy.
    The function will search in the criterias trying to find one criteria with same Name/Id passed as parameter, if any criteria match with the parameter, the function'll return NULL, otherwise the function will call the Criteria constructor and then add this memory address in the criterias vector.
        \param The variable name/id.
        \return The criteria pointer or NULL.
 */
Node* Hierarchy::addCriteria(const char* name) {
	//Iterate though criterias array.
	int i;
	for ( i = 0; i < this->criterias_size; i++) {
		//If found some criteria with same name/id return NULL
		if (strcmp(this->criterias[i]->getName(), name) == 0) {
			return NULL;
		}
	}
	//Call the criteria's constructor (i.e., create a new criteria)
	Node* c = new Node();
	c->setName(name);
	c->setTypeCriteria();
	//Add the criteria in hierarchy (criterias vector).
	this->criterias = (Node**) realloc (this->criterias, sizeof(Node*) * this->criterias_size+1);
	this->nodes = (Node**) realloc (this->nodes, sizeof(Node*)* this->nodes_size+1);

	this->nodes[this->nodes_size] = c;
	this->criterias[this->criterias_size] = c;

	this->nodes_size++;
	this->criterias_size++;
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
void Hierarchy::addEdge(const char* parentName, const char* childName, float* weight, int size) {
	//Search for some criteria with same name/id.
	Node *parent = findCriteria(parentName);
	//Search for some criteria with same name/id.
	Node *child = findCriteria(childName);
	//if one of the two nodes not exists in the hierarchy exit this function.
	if (parent == NULL || child == NULL) {
		return;
	}
	//Call the Edge constructor (i.e., create a new edge)
	Edge *edge = new Edge(child, weight, size);
	//Add the edge address in the hierarchy (i.e., edges vector)
	parent->addEdge(edge);
}


/**
    \brief Add an edge between Focus and Criteria nodes.
    \param objective: Focus pointer.
    \param child: Criteria pointer.
    \param w: the edge weight. Default 0.
 */
void Hierarchy::addEdgeObjective(Node *objective, Node *child, float* weight, int size) {
	//Call the edge constructor (i.e., creates a new edge).
	Edge *edge = new Edge(child, weight, size);
	//Add this edge in the Focus edges list.
	this->objective->addEdge(edge);
}

/**
   \brief Add an edge between two Criteria nodes.
   \param parent: Criteria pointer as father.
   \param child: Criteria pointer as child.
   \param w: the edge weight. Default 0.
 */
void Hierarchy::addEdgeCriteria(Node *parent, Node *child, float* weight, int size){
	//Call the edge constructor (i.e., create a new edge).
	Edge *edge = new Edge(child, weight, size);
	//Add this edge in the father edges list.
	parent->addEdge(edge);
}

/**
   \brief Add an edge between Criteria and Alternative nodes.
   \param parent: Criteria pointer.
   \param alt: Alternative pointer.
   \param w: the edge weight. Default 0.
 */
void Hierarchy::addEdgeAlternative(Node *parent, Node *alt, float* weight, int size) {
	//Call the edge constructor (i.e., create new edge).
	Edge *edge = new Edge(alt, weight, size);
	//Add the edge in the parent edges list.
	parent->addEdge(edge);
}

/**
    \brief Add one resource in the hierarchy default resources.
    This function will add new resource in the hierarchy default resources, float type are used.
    \param name: The resource name.
 */
void Hierarchy::addResource(char* name) {
	this->resource.addResource(name, 0);
}

/**
    \brief Add an alternative pointer in the hierarchy.
    \return An alternative pointer.
 */
Node* Hierarchy::addAlternative() {
	//Call the node constructor (i.e., new alternative).
	Node *alternative = new Node();
	alternative->setTypeAlternative();
	alternative->setResource(this->resource);

	//Add the alternative pointer in the hierarchy (i.e., the alternatives vector).
	this->alternatives = (Node**) realloc (this->alternatives, sizeof(Node*)* this->alternatives_size+1);
	this->nodes = (Node**) realloc (this->nodes, sizeof(Node*)*this->nodes_size+1);

	this->alternatives[this->alternatives_size] = alternative;
	this->nodes[this->nodes_size] = alternative;

	this->alternatives_size++;
	this->nodes_size++;

	return alternative;
}

/**
    \brief Add an alternative pointer in the hierarchy.
    \return An alternative pointer.
 */
Node* Hierarchy::addAlternative(Node* alternative) {
	//Add the alternative pointer in the hierarchy (i.e., the alternatives vector).
	this->alternatives = (Node**) realloc (this->alternatives, sizeof(Node*)* this->alternatives_size+1);
	this->nodes = (Node**) realloc (this->nodes, sizeof(Node*)*this->nodes_size+1);

	this->alternatives[this->alternatives_size] = alternative;
	this->nodes[this->nodes_size] = alternative;

	this->alternatives_size++;
	this->nodes_size++;
	return alternative;
}

/**
    \brief Add an edge between the hierarchy sheets and the alternatives.
 */
void Hierarchy::addEdgeSheetsAlternatives() {
	int i,j;
	//Iterate through all the sheets.
	for(i=0; i<this->sheets_size; i++) {
		//Iterate through all the alternatives.
		for(j=0; j<this->alternatives_size; j++) {
			//Create an edge between them.
			this->addEdgeAlternative(this->sheets[i], this->alternatives[j]);
		}
	}
}

/**
    \brief Add a new sheet.
    This function will add the existing criteria in the sheets vector if this criteria has the leaf value set as true.
 */
void Hierarchy::addSheets(Node* criteria) {
	//If the criteria isn't a sheet.
	if (!findSheets(criteria)) {
		//Add the criteria in the sheets vector.
		this->sheets = (Node**) realloc (this->sheets, sizeof(Node*)* this->sheets_size+1);

		this->sheets[this->sheets_size+1] = criteria;

		this->sheets_size++;
	}
}

/*Printing status Function*/

/**
    \brief List the main objective name.
 */
void Hierarchy::listFocus() {
	//If the objective exists, print their name.
	if (this->objective != NULL) {
		printf("Focus %s\n", this->objective->getName());
	}else{
		printf("Empty Focus\n");
	}
}

/**
   \brief List all the criterias names in the hierarchy.
 */
void Hierarchy::listCriteria() {
	int i;
	//Iterate through the criterias list and print their names.
	for(i=0; i<this->criterias_size; i++) {
		printf("%s\n", this->criterias[i]->getName());
	}
}

/**
    \brief List all the resources in the default resource of hierarchy.
    The function will check the size of each map and if their size is bigger than 0, the variable name and value will be printed.
 */
void Hierarchy::listResources() {
	int i;
	int size = this->resource.getDataSize();
	for(i=0; i<size; i++) {
		printf("%s: %lf\n",  this->resource.getResourceName(i), this->resource.getResource(i));
	}
}

/*Finding Functions*/

/**
    \brief Search for an specific criteria in the criterias list.
    \param c: Criteria point to be sought.
    \return True if the criteria was found, false otherwise.
 */
bool Hierarchy::findCriteria(Node *c) {
	int i;
	//iterate through the criterias in the hierarchy.
	for ( i=0; i<nodes_size; i++) {
		if ( (this->nodes[i]->getName(), c->getName() ) ==0) {
			return true;
		}
	}
	return false;
}

/**
    \brief Search for an specific criteria thorught its name/id.
    \param name: The name/id to search.
    \return The criteria pointer if exists one criteria with the name parameter, NULL otherwise.
 */
Node* Hierarchy::findCriteria(const char* name) {
	int i;
	//iterate through the criterias in the hierarchy.
	for ( i=0; i<this->nodes_size; i++) {
		if ( (this->nodes[i]->getName(), name ) ==0) {
			return this->nodes[i];
		}
	}
	return NULL;
}

/**
    \brief  The function will search the sheets list trying to find the a specific criteria in the list.
    \param c: Criteria pointer.
    \return True if the criteria are in the sheets list, false otherwise.
 */
bool Hierarchy::findSheets(Node *c) {
	int i;
	//Iterate through all the sheets list.
	for ( i=0; i<this->sheets_size; i++) {
		if ( strcmp( this->sheets[i]->getName(),  c->getName())==0) {
			return true;
		}
	}
	return false;
}

/**
    \brief The function will search the alternatives list trying to find an alternative with the respective param name/id.
    \param v: The alternative name/id to find.
    \return Alternative pointer if exista an alternative with the respective name/id, NULL otherwise.
 */
Node * Hierarchy::findAlternative(const char* name) {
	int i;
	for ( i=0; i < this->alternatives_size; i++) {
		if ( strcmp(this->alternatives[i]->getName(), name) ==0 ) {
			return (this->alternatives[i]);
		}
	}
	return NULL;
}

/*Clear Functions*/

/**
    \brief The function will clear all the edges between the sheets and alternatives.
 */
void Hierarchy::clearSheetsEdges() {
	int i;
	int size = this->sheets_size;
	//Iterate through all the sheets.
	for ( i=0; i<size; i++) {
		this->sheets[i]->clearEdges();
	}
	free(this->sheets);
	this->sheets = NULL;
	this->sheets_size = 0;
}

/**
    \brief The function will clear all the alternatives in the hierarchy.
 */
void Hierarchy::clearAlternatives() {
	int i=0;
	int size= this->alternatives_size;
	for(i=0; i<size; i++) {
		delete(this->alternatives[i]);
	}
	free(this->alternatives);
	this->alternatives = NULL;
	this->alternatives_size=0;
	clearSheetsEdges();
}

void Hierarchy::clearResource(){
	this->resource.clear();
}

/*Update Functions*/

/**
    \brief The function update all the edges between the sheets list and the criterias.
 */
void Hierarchy::updateSheetsEdges() {
	clearSheetsEdges();
	addEdgeSheetsAlternatives();
}

/*Getting Focus*/

/**
    \brief The function return the main objective pointer.
    \return The focus pointer
 */
Node* Hierarchy::getFocus() {
	return this->objective;
}

/**
    \brief Return the size of the nodes list.
    \return Nodes size.
 */
int Hierarchy::getNodesSize(){
	return this->nodes_size;
}

/**
    \brief Return the size of the sheets list.
    \return Sheets size.
 */
int Hierarchy::getSheetsSize() {
	return this->sheets_size;
}

/**
    \brief Return the size of the alternatives list.
    \return Alternatives size.
 */
int Hierarchy::getAlternativesSize()  {
	return this->alternatives_size;
}

/**
    \brief Get function to return the resource pointer.
    \return Return resource pointer.
 */
H_Resource* Hierarchy::getResource() {
	return &(this->resource);
}

/**
    \brief Get function to return the criteria pointer.
    \return Return criteria pointer.
 */
Node** Hierarchy::getCriterias() {
	return this->criterias;
}

/**
    \brief Get function to return the sheets list.
    \return Return sheets list.
 */
Node** Hierarchy::getSheets() {
	return this->sheets;
}

/**
    \brief Get function to return the alternatives list.
    \return Return alternatives list.
 */
Node** Hierarchy::getAlternatives() {
	return this->alternatives;
}
