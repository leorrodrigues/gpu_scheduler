#ifndef _INTERVAL_TREE_
#define _INTERVAL_TREE_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stack>

#include "../thirdparty/spdlog/spdlog.h"

namespace Interval_Tree {

typedef struct node_interval_t {
	int low;
	int high;
	float capacity;
} node_interval_t;

typedef struct interval_t {
	int size;
	node_interval_t *nodes;

	void clear(){
		free(nodes);
		nodes = NULL;
	}
} interval_t;

typedef struct node_t {
	unsigned int size; // the total ammount of children the node has.
	int interval[2]; // The interval that the node has [_,-1] to define infinity
	int height; // The height of the subtree

	int max;
	float capacity; // The total ammount of capacity the current node has available

	node_t *left;
	node_t *right;
} node_t;

class Interval_Tree {

private:
typedef struct {
	node_t *root;
	float capacity; // the capacity of the server
} interval_tree_t;

interval_tree_t *tree;

inline int getHeight(node_t* node){
	return node==NULL ? -1 : node->height;
}

inline int getSize(node_t* node){
	return node==NULL ? 0 : node->size+1;
}

inline int getMax(int a, int b){
	return a>b ? a : b;
}

inline int getMin(int a, int b){
	return a>b ? b : a;
}

inline int getBF(node_t* node){
	return getHeight(node->right)-getHeight(node->left);
}

inline void single_rotate_left(node_t* x){
	node_t *temp =  x->right;

	std::swap((*x),((*x->right)));

	temp->right = x->left;
	x->left = temp;

	x->left->height = getMax(getHeight(x->left->left),getHeight(x->left->right)) + 1;

	x->height = getMax(getHeight(x->left),getHeight(x->right)) + 1;

	x->left->size = getSize(x->left->left) + getSize(x->left->right);
	x->size = getSize(x->left) + getSize(x->right);
}

inline void single_rotate_right(node_t* x){
	node_t *temp =  x->left;

	std::swap((*x),((*x->left)));

	temp->left = x->right;
	x->right = temp;

	x->right->height = getMax(getHeight(x->right->left),getHeight(x->right->right)) + 1;
	x->height = getMax(getHeight(x->left),getHeight(x->right)) + 1;

	x->right->size = getSize(x->right->left) + getSize(x->right->right);
	x->size = getSize(x->left) + getSize(x->right);
}

void show_bf(node_t *node,unsigned int tab){
	unsigned int i=0;
	if(node==NULL) {
		for(i=0; i<tab; i++)
			spdlog::debug("\t");
		spdlog::debug("Node: NULL\n");
		return;
	}
	for(i=0; i<tab; i++)
		spdlog::debug("\t");
	spdlog::debug("Node in [{},{}] # Elements: {} # Height {} # Capacity: {}\n", node->interval[0], node->interval[1], node->size, node->height, node->capacity);
	show_bf(node->left,tab+1);
	show_bf(node->right,tab+1);
}

void free_nodes(node_t *node){
	if(node == NULL) return;
	free_nodes(node->left);
	free_nodes(node->right);
	free(node);
	node=NULL;
}

void copyTree(Interval_Tree *source){
	int size = source->tree->root->size+1, index=-1, aux=0;
	node_t** queue = (node_t**)calloc(size, sizeof(node_t*)); //malloc the tree's size (worst case).

	queue[0] = source->tree->root;
	while(++index < size) {
		if(queue[index]->left != NULL)
			queue[++aux] = queue[index]->left;

		if(queue[index]->right != NULL)
			queue[++aux] = queue[index]->right;

		insert(queue[index]->interval[0], queue[index]->interval[1], queue[index]->capacity);
	}
	free(queue);
	queue=NULL;
}

inline void copyToArray(interval_t *dst){
	std::stack<node_t*> stack;
	node_t *current = tree->root;
	int index=-1;
	while(true) {
		while(current!=NULL) {
			stack.push(current);
			current = current->left;
		}
		if(stack.empty()) return;
		current = stack.top();
		stack.pop();
		dst->nodes[++index].low = current->interval[0];
		dst->nodes[index].high = current->interval[1];
		dst->nodes[index].capacity = current->capacity;
		current = current->right;
	}
}

inline bool hasOverlap(int low1, int high1, int low2, int high2){
	if ((low1 < high2 && low2 < high1) || (low1 > high2 && low2 > high1)) {
		return true;
	}
	return false;
}

inline node_t *overlapSearch(node_t *start, node_t *node) {
	int index=-1,aux=-1, size = start->size+1;
	node_t** queue = (node_t**)calloc(size, sizeof(node_t*)); //malloc the tree's size (worst case).
	if(start->interval[0] == node->interval[0] && start->interval[1] == node->interval[1]) {
		if(node->left!=NULL)
			queue[++aux] = node->left;
		if(node->right!=NULL)
			queue[++aux] = node->right;
	}else{
		queue[++aux] = start;
	}

	while(++index < size) {
		if(queue[index] != NULL) {
			if(queue[index]->left!=NULL && node->interval[0] <= queue[index]->left->max)
				queue[++aux] = queue[index]->left;

			if(queue[index]->right!=NULL && node->interval[0] <= queue[index]->right->max) queue[++aux] = queue[index]->right;

			if (queue[index]->interval[0] == node->interval[0] && queue[index]->interval[1] == node->interval[1]) continue;

			if( hasOverlap(node->interval[0], node->interval[1], queue[index]->interval[0], queue[index]->interval[1])) {
				node_t *temp = queue[index];
				free(queue);
				return temp;
			}
		}
	}
	free(queue);
	return NULL;
}

public:
Interval_Tree(float capacity = 0){
	if((tree = (interval_tree_t*)calloc(1, sizeof(interval_tree_t)))==NULL) {
		SPDLOG_ERROR("Error in malloc\n");
		exit(1);
	}
	tree->root=NULL;
	tree->capacity = capacity;
}

~Interval_Tree(){
	free_nodes(tree->root);
	free(tree);
	tree=NULL;
}

void show(){
	spdlog::debug("The Bottleneck Interval Tree ");
	if(tree->root==NULL) {
		spdlog::debug("don't has any elements.\n");
	}else{
		spdlog::debug("has {} elements.\n", tree->root->size+1);
		show_bf(tree->root, 0);
	}
}

void setCapacity(float capacity){
	tree->capacity = capacity;
}

float getCapacity(){
	return tree->capacity;
}

void insert(int interval_low, int interval_high, float capacity){
	if(capacity > tree->capacity) {
		SPDLOG_ERROR("The interval can't consume more capacity than the tree capacity ![{} > {}]\n", capacity, tree->capacity);
		exit(0);
	}
	node_t **path = NULL;
	int index=-1;
	int key = interval_low;

	//allocate the paths size
	if(tree->root != NULL)
		path = (node_t**)calloc((tree->root->height+3), sizeof(node_t*)); //allocate the total ammount needed to build the path

	//iterate through the tree to get the nodes
	{
		node_t *aux = tree->root;

		if(tree->root != NULL && aux->interval[0] == interval_low)
			path[++index]=tree->root;

		while(aux != NULL && aux->interval[0] != interval_low) {        //while not found an empty child and the current node dont has the same low interval of the new node
			++index;
			path[index] = aux;
			aux = interval_low > aux->interval[0] ? aux->right : aux->left;
		}

		if(aux!=NULL) {
			//iterate through the nodes that has the same low interval to check the hight interval. If exists a node with the same [a,b] interval, just add the total ammount of
			while(aux->interval[1] != interval_high) {
				++index;
				path[index] = aux;
				if(aux!=NULL) break;
				if(interval_high >= aux->interval[1]) {
					aux = aux->right;
				} else {
					aux = aux->left;
				}

			}
		}

		if(aux!=NULL && aux->interval[0] == interval_low && aux->interval[1] == interval_high) {
			aux->capacity+=capacity;
			free(path);
			path=NULL;
			if(aux->capacity > tree->capacity) {
				SPDLOG_ERROR("The interval can't consume more capacity than the tree capacity\n");
				exit(0);
			}
			return;
		}

	}

	node_t *new_node;
	if((new_node = (node_t*)calloc(1, sizeof(node_t)))==NULL) {
		SPDLOG_ERROR("Error in alocate new node\n");
		exit(1);
	}

	new_node->left = NULL;
	new_node->right = NULL;
	new_node->size = 0;
	new_node->height=0;

	new_node->max = interval_high;
	new_node->interval[0] = interval_low;
	new_node->interval[1] = interval_high;

	new_node->capacity = capacity;

	if(index==-1) {
		tree->root=new_node;
	}else{
		if(key >= path[index]->interval[0])
			path[index]->right = new_node;
		else
			path[index]->left = new_node;

		++path[index]->size;
		path[++index]=new_node;
	}
	int bf;
	while(index>=0) {         // returning the path, updating the nodes' height and check if is needed to make some rotation.
		//updating the height
		path[index]->height = getMax(getHeight(path[index]->left),getHeight(path[index]->right)) + 1;

		path[index]->size = getSize(path[index]->left)+getSize(path[index]->right);

		if(index>0)
			path[index]->max = getMax(path[index-1]->max, path[index]->max);

		//check the balancing factor
		bf = getBF(path[index]);

		if(bf > 1 && key > path[index]->right->interval[0]) {         // single left
			single_rotate_left(path[index]);
			bf = getBF(path[index]);
		}
		if(bf < -1 && key < path[index]->left->interval[0]) {         // single right
			single_rotate_right(path[index]);
			bf = getBF(path[index]);
		}
		if(bf > 1 && key < path[index]->right->interval[0]) {         // double left
			single_rotate_right(path[index]->right);
			single_rotate_left(path[index]);
			bf = getBF(path[index]);
		}
		if(bf < -1 && key > path[index]->left->interval[0]) {         // double right
			single_rotate_left(path[index]->left);
			single_rotate_right(path[index]);
			bf = getBF(path[index]);
		}
		--index;
	}
	if(path!=NULL)
		free(path);
	path=NULL;
}

void remove(int key, int s_key, float capacity = 0){        //key = low_interval, s_key = high_interval
	node_t **path = NULL;
	if(tree->root == NULL)         //empty tree
		return;
	else
		path = (node_t**)calloc((tree->root->height+2), sizeof(node_t*));         //allocate the total ammount needed to build the path, it's a ternary pointer to store all the original pointers of Interval_Tree

	int index=-1;

	{
		node_t *aux = tree->root;
		while(aux != NULL) {        //while not found a empty child
			++index;
			path[index] = aux;         // update the path
			if(aux->interval[0] == key) {         // equal keys aren't allowed
				break;         // find the node
			}
			aux = key > aux->interval[0] ? aux->right : aux->left;
		}
		while(aux != NULL && aux->interval[1] != s_key) {
			aux = s_key > aux->interval[1] ? aux->right : aux->left;
		}
		if(aux==NULL) {         // didn't find any child that hold the key for deletion
			free(path);         // Free the parh variable
			path=NULL;
			spdlog::debug("Don't existis any node that holds the specific interval [{};{}]\n", key, s_key);
			return;
		}else if(aux->interval[1] == s_key) {
			aux->capacity-=capacity;
			if(aux->capacity<0) {
				SPDLOG_ERROR("Error Negative Capacity\n");
				free(path);
				path=NULL;
				exit(0);
			}else if(aux->capacity > 0) {
				free(path);
				path=NULL;
				return;
			}
		}
	}

	int index_sub = index;

	node_t *temp; // = (node_t*) malloc(sizeof(node_t));
	if (path[index_sub]->size>1) { // the node two children
		temp = path[index_sub]->left;
		path[index_sub+1] = path[index_sub]->left;
		++index_sub;
		while(temp->right != NULL) { // get the node with the largest key in the left subtree of the selected node
			path[index_sub+1] = path[index_sub]->right;
			++index_sub;
			temp = temp->right;
		}

		if(index>0) {
			if(path[index]->right != temp)
				temp->right = path[index]->right;

			if(path[index]->left != temp)
				temp->left = path[index]->left;
			if(path[index-1]->left->interval[0]>temp->interval[0])
				path[index-1]->left = temp;
			else
				path[index-1]->right = temp;
			if(path[index_sub-1]->left==path[index_sub])
				path[index_sub-1]->left = NULL;
			else
				path[index_sub-1]->right = NULL;
			free(path[index]);
			path[index] = temp;
			path[index_sub] = NULL;
		}else{
			if(temp != path[index]->right)
				temp->right = path[index]->right;
			if(temp != path[index]->left)
				temp->left = path[index]->left;
			if(path[index_sub-1]->left == path[index_sub])
				path[index_sub-1]->left = NULL;
			else
				path[index_sub-1]->right = NULL;
			free(tree->root);
			path[index] = temp;
			tree->root = temp;
		}
	} else { // the node has at least one child
		if(path[index_sub]->left==NULL && path[index_sub]->right==NULL) {         //no child
			if(index_sub>0) {
				if(path[index_sub-1]->left == path[index_sub]) {
					free(path[index_sub-1]->left);
					path[index_sub-1]->left = NULL;
				} else if(path[index_sub-1]->right == path[index_sub]) {
					free(path[index_sub-1]->right);
					path[index_sub-1]->right = NULL;
				}
				path[index_sub] = NULL;
			}else{
				free(tree->root);
				tree->root=NULL;
				path[index] = NULL;
			}
		} else{         //one child
			if(index_sub>0) {
				if(path[index_sub]->left != NULL) {
					temp = path[index_sub]->left;
				} else {
					temp = path[index_sub]->right;
				}
				if(path[index_sub-1]->left == path[index_sub]) {
					free(path[index_sub-1]->left);
					path[index_sub-1]->left = temp;
				}else{
					free(path[index_sub-1]->right);
					path[index_sub-1]->right = temp;
				}
				path[index_sub] = temp;
			}else{
				if(path[index]->left != NULL) {
					temp = path[index]->left;
				} else {
					temp = path[index]->right;
				}
				free(tree->root);
				tree->root = temp;
				path[index] = temp;
			}
		}
	}

	index = index_sub;         // remove the last element index
	int bf;
	while(index>=0) {         // returning the path, updating the nodes' height and check if is needed to make some rotation.
		//updating the height
		if(path[index]==NULL) {
			--index;
			continue;
		}

		path[index]->height = getMax(getHeight(path[index]->left),getHeight(path[index]->right)) + 1;

		path[index]->size = getSize(path[index]->left)+getSize(path[index]->right);

		if(index>0)
			path[index]->max = getMax(path[index-1]->max, path[index]->max);

		//check the balancing factor
		bf = getBF(path[index]);

		if(bf > 1 && getBF(path[index]->right) >= 0) {         // single left
			single_rotate_left(path[index]);
			bf = getBF(path[index]);
		}
		if(bf < -1 && getBF(path[index]->left) < 0) {         // single right
			single_rotate_right(path[index]);
			bf = getBF(path[index]);
		}
		if(bf > 1 && getBF(path[index]->right) < 0) {         // double left
			single_rotate_right(path[index]->right);
			single_rotate_left(path[index]);
			bf = getBF(path[index]);
		}
		if(bf < -1 && getBF(path[index]->left) >= 0) {         // double right
			single_rotate_left(path[index]->left);
			single_rotate_right(path[index]);
			bf = getBF(path[index]);
		}
		index--;
	}
	if(path!=NULL)
		free(path);
	path=NULL;
}

interval_t* getInterval(int p_key, int s_key){// retorna um vetor com todos os intervalos e seus consumos de recursos
	Interval_Tree result_tree(tree->capacity);
	{
		Interval_Tree aux_tree(tree->capacity);
		//At first, it is needed to transform the tree to make the tree contain only nodes that overlap the interval [p_key, s_key].
		{
			int index=-1,aux=0, size = tree->root->size+1, low, high;
			node_t** queue = (node_t**)calloc(size, sizeof(node_t*)); //malloc the tree's size (worst case).
			queue[0] = tree->root;

			while(++index < size) {
				if(queue[index] == NULL)
					break;

				if( hasOverlap(p_key, s_key, queue[index]->interval[0], queue[index]->interval[1])) {
					low = queue[index]->interval[0]  < p_key ? p_key :  queue[index]->interval[0];

					high = queue[index]->interval[1]  > s_key ? s_key :  queue[index]->interval[1];
					aux_tree.insert(low, high, queue[index]->capacity);
				}

				if(queue[index]->left!=NULL && p_key <= queue[index]->left->max)
					queue[++aux] = queue[index]->left;

				if(queue[index]->right!=NULL && p_key <= queue[index]->right->max)
					queue[++aux] = queue[index]->right;

			}
			free(queue);
			if(aux_tree.tree->root == NULL) // the interval don't overlap any node. Just return NULL.
				return NULL;
		}
		// After the new tree is built, merge the nodes that contain some type of overlap.
		{
			int index=0,aux=0, size = aux_tree.tree->root->size+1, low, high;
			float c, c_temp;
			node_t** queue = (node_t**)calloc(size+3, sizeof(node_t*)); //malloc the tree's size (worst case).
			node_t *temp = NULL;
			queue[index] = aux_tree.tree->root;

			while((unsigned int)index < aux_tree.tree->root->size+1) {
				if(queue[index] == NULL) {
					continue;
				}
				temp = overlapSearch(aux_tree.tree->root, queue[index]);
				if(temp==NULL) {        // if the current node don't has any other node that contain an overlaping
					//Insert the children in queue
					if(queue[index]->left!=NULL && p_key <= queue[index]->left->max)
						queue[++aux] = queue[index]->left;

					if(queue[index]->right!=NULL && p_key <= queue[index]->right->max)
						queue[++aux] = queue[index]->right;

					// Update the index to get the next node
					++index;
				}else{ // the current node has overlaping, update the limits and capacities.
					if (queue[index]->interval[0] != temp->interval[0] && queue[index]->interval[1] != temp->interval[1]) { // the merge will create a new node
						// Updating the nodes
						low = getMin(queue[index]->interval[0], temp->interval[0]);
						queue[index]->interval[0] = getMax(queue[index]->interval[0], temp->interval[0]);
						temp->interval[0] = low;

						high = getMax(queue[index]->interval[1], temp->interval[1]);

						queue[index]->interval[1] = getMin(queue[index]->interval[1], temp->interval[1]);
						temp->interval[1] = queue[index]->interval[0];

						low = queue[index]->interval[1];
						c_temp = temp->capacity;
						c = queue[index]->capacity;
						queue[index]->capacity += temp->capacity;
						temp->capacity = c;
						aux_tree.insert(low, high, c_temp);
					}else if(queue[index]->interval[0] == temp->interval[0]) { // only update the current nodes
						if(queue[index]->interval[1] > temp->interval[1]) {
							queue[index]->capacity += temp->capacity;
						}else{
							c = temp->capacity;
							temp->capacity += queue[index]->capacity;
							queue[index]->capacity = c;
						}
						// Updating the nodes
						high = getMax(queue[index]->interval[1], temp->interval[1]);

						queue[index]->interval[0] = getMin(queue[index]->interval[1], temp->interval[1]);

						queue[index]->interval[1] = high;
						temp->interval[1] = queue[index]->interval[0];

					}else if(queue[index]->interval[1] == temp->interval[1]) {
						if(queue[index]->interval[0] > temp->interval[0]) {
							queue[index]->capacity += temp->capacity;
						}else{
							c = queue[index]->capacity;
							queue[index]->capacity += temp->capacity;
							temp->capacity = c;
						}

						// Updating the nodes
						low = getMin(queue[index]->interval[0], temp->interval[0]);

						temp->interval[1] = getMax(queue[index]->interval[0], temp->interval[0]);
						temp->interval[0] = low;
						queue[index]->interval[0] = temp->interval[1];
					}
				}
			}
			free(queue);
		}
		result_tree.copyTree(&aux_tree);
	}
	//Finally, when the merge of all nodes were done, convert the tree into a sorted array. To do that, get the left subtree first, then the parent node and then the right subtree.

	interval_t *result = (interval_t*) calloc (1, sizeof(interval_t));

	result->size = result_tree.tree->root->size+1;
	result->nodes = (node_interval_t*) calloc (result->size, sizeof(node_interval_t));

	result_tree.copyToArray(result);

	// Return the array.
	return result;
}

inline float getMinValueAvailable(int p_key, int s_key){// retorna um float com a menor capacidade do intervalo
	interval_t *interval = getInterval(p_key, s_key);
	float capacity = 0;

	if(interval==NULL) return capacity;

	for(int i=0; i < interval->size; i++)
		capacity = getMax(capacity, interval->nodes[i].capacity);

	interval->clear();
	free(interval);
	interval = NULL;

	return (tree->capacity - capacity);
}

Interval_Tree& operator+= (Interval_Tree& rhs){
	int index=-1,aux=0, size = rhs.tree->root->size+1;
	node_t** queue = (node_t**)calloc(size, sizeof(node_t*));         //malloc the tree's size (worst case).
	this->tree->capacity += rhs.tree->capacity;
	queue[0] = rhs.tree->root;

	while(++index < size) {
		if(queue[index] == NULL)
			break;

		this->insert(queue[index]->interval[0], queue[index]->interval[1], queue[index]->capacity);

		if(queue[index]->left!=NULL)
			queue[++aux] = queue[index]->left;

		if(queue[index]->right!=NULL)
			queue[++aux] = queue[index]->right;

	}
	free(queue);
	return *this;
}

Interval_Tree& operator-= (Interval_Tree& rhs){
	int index=-1,aux=0, size = rhs.tree->root->size+1;
	node_t** queue = (node_t**)calloc(size, sizeof(node_t*));         //malloc the tree's size (worst case).
	if(rhs.tree->capacity > this->tree->capacity) {
		SPDLOG_ERROR("The tree can't has capacity less than 0\n");
		return *this;
	}
	this->tree->capacity -= rhs.tree->capacity;
	queue[0] = rhs.tree->root;

	while(++index < size) {
		if(queue[index] == NULL)
			break;

		this->remove(queue[index]->interval[0], queue[index]->interval[1], queue[index]->capacity);

		if(queue[index]->left!=NULL)
			queue[++aux] = queue[index]->left;

		if(queue[index]->right!=NULL)
			queue[++aux] = queue[index]->right;

	}
	free(queue);
	return *this;
}

};

}
#endif
