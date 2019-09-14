#ifndef _INTERVAL_TREE_
#define _INTERVAL_TREE_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stack>
#include <cmath>

#include "../thirdparty/spdlog/spdlog.h"

namespace Interval_Tree {

typedef struct node_interval_t {
	int low;
	int high;
	double capacity;
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

	double capacity; // The total ammount of capacity the current node has available

	node_t *left;
	node_t *right;
} node_t;

class Interval_Tree {

private:
typedef struct {
	node_t *root;
	double capacity; // the capacity of the server
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

inline node_t *single_rotate_left(node_t* actual){
	node_t *temp =  actual->right;
	node_t *aux = temp->left;

	temp->left = actual;
	actual->right = aux;

	actual->height = getMax(getHeight(actual->left),getHeight(actual->right)) + 1;
	temp->height = getMax(getHeight(temp->left),getHeight(temp->right)) + 1;

	actual->size = getSize(actual->left) + getSize(actual->right);
	temp->size = getSize(temp->left) + getSize(temp->right);

	return temp;
}

inline node_t *single_rotate_right(node_t* actual){
	node_t *temp =  actual->left;
	node_t *aux = temp->right;

	temp->right = actual;
	actual->left = aux;

	actual->height = getMax(getHeight(actual->left),getHeight(actual->right)) + 1;
	temp->height = getMax(getHeight(temp->left),getHeight(temp->right)) + 1;

	actual->size = getSize(actual->left) + getSize(actual->right);
	temp->size = getSize(temp->left) + getSize(temp->right);

	return temp;
}

void show_bf(node_t *node,unsigned int tab){
	unsigned int i=0;
	std::string tab_str;
	if(node==NULL) {
		for(i=0; i<tab; i++)
			tab_str += "\t";
		spdlog::debug("{}Node: NULL", tab_str);
		return;
	}
	tab_str = "";
	for(i=0; i<tab; i++)
		tab_str += "\t";
	spdlog::debug("{}Node in [{},{}] # Elements: {} # Height {} # Capacity: {}", tab_str, node->interval[0], node->interval[1], node->size, node->height, node->capacity);
	show_bf(node->left,tab+1);
	show_bf(node->right,tab+1);
}

inline void free_nodes(node_t *node){
	if(node == NULL) return;
	free_nodes(node->left);
	free_nodes(node->right);
	free(node);
	node=NULL;
}

inline void copyTree_iteration(node_t *node){
	if(node == NULL) return;
	insert(node->interval[0], node->interval[1], node->capacity);
	copyTree_iteration(node->left);
	copyTree_iteration(node->right);
}

inline void copyTree(Interval_Tree *src){
	copyTree_iteration(src->tree->root);
}

inline void copyToArray_interation(interval_t *dst, int *index, node_t *node){
	if(node == NULL) return;

	copyToArray_interation(dst, index, node->left);
	copy_node(&dst->nodes[*index], node);
	++(*index);
	copyToArray_interation(dst, index, node->right);
}

inline void copyToArray(interval_t *dst){
	int index = 0;
	if(tree->root != NULL)
		copyToArray_interation(dst, &index, tree->root);
}

inline bool hasOverlap(int low1, int high1, int low2, int high2){
	if(low1 == low2 && high1 == high2) return false;
	if(low1 < high2 && low2 < high1)
		return true;
	return false;
}

inline bool hasOverlapShrink(int low1, int high1, int low2, int high2){
	if(low1 <= high2 && low2 <= high1)
		return true;
	return false;
}

inline node_t *overlapSearch(node_t *node, node_t *actual) {
	if(node == NULL || actual == NULL) return NULL;

	if(!hasOverlap(node->interval[0], node->interval[1], actual->interval[0], actual->interval[1])) {
		node_t *temp = NULL;
		temp = overlapSearch(node, actual->left);
		if(temp != NULL) return temp;
		temp = overlapSearch(node, actual->right);
		if(temp != NULL) return temp;
		return NULL;
	}
	return actual;
}

inline node_t *create_node(int low, int high, double capacity){
	node_t *new_node;

	if((new_node = (node_t*)calloc(1, sizeof(node_t)))==NULL) {
		SPDLOG_ERROR("Error in alocate new node");
		exit(1);
	}

	new_node->left = NULL;
	new_node->right = NULL;
	new_node->size = 0;
	new_node->height=0;
	new_node->interval[0] = low;
	new_node->interval[1] = high;
	new_node->capacity = capacity;

	return(new_node);
}

inline void copy_node(node_interval_t *dst, node_t *src){
	dst->low = src->interval[0];
	dst->high = src->interval[1];
	dst->capacity = src->capacity;
}

inline void copy_node(node_t *dst, node_t *src){
	dst->interval[0] = src->interval[0];
	dst->interval[1] = src->interval[1];
	dst->capacity = src->capacity;
}

inline node_t *getMaxSubtree(node_t *node){
	node_t *temp = node;
	while(temp->right != NULL) {
		temp = temp->right;
	}
	return temp;
}

inline node_t *getMinSubtree(node_t *node){
	node_t *temp = node;
	while(temp->left != NULL) {
		temp = temp->left;
	}
	return temp;
}

inline node_t *insert_iteration(node_t *actual, node_t *new_node){
	if(actual == NULL)
		return new_node;

	if(new_node->interval[0] < actual->interval[0]) {
		actual->left = insert_iteration(actual->left, new_node);
	} else if(new_node->interval[0] > actual->interval[0]) {
		actual->right = insert_iteration(actual->right, new_node);
	} else{ // the nodes has the same low interval
		if(new_node->interval[1] < actual->interval[1]) {
			actual->left = insert_iteration(actual->left, new_node);
		} else if(new_node->interval[1] > actual->interval[1]) {
			actual->right = insert_iteration(actual->right, new_node);
		} else{ // the nodes has the same low and high intervals
			actual->capacity += new_node->capacity;
			free(new_node);
			return actual;
		}
	}

	//updating the height
	actual->height = getMax(getHeight(actual->left),getHeight(actual->right)) + 1;

	actual->size = getSize(actual->left)+getSize(actual->right);

	//check the balancing factor
	int bf = getBF(actual);

	int index = 0; //0 to represent the low interval and 1 to high interval
	if(new_node->interval[0] == actual->interval[0]) { // otherwise, the low intervals are the same, so, rebalance the nodes thorugh the high interval
		index = 1;
	}

	if(bf > 1 && new_node->interval[index] > actual->right->interval[index]) { // single left
		return single_rotate_left(actual);
	}
	if(bf < -1 && new_node->interval[index] < actual->left->interval[index]) { // single right
		return single_rotate_right(actual);
	}
	if(bf > 1 && new_node->interval[index] < actual->right->interval[index]) { // double left
		actual->right = single_rotate_right(actual->right);
		return single_rotate_left(actual);
	}
	if(bf < -1 && new_node->interval[index] > actual->left->interval[index]) { // double right
		actual->left = single_rotate_left(actual->left);
		return single_rotate_right(actual);
	}

	return actual;
}

inline node_t *remove_iteration(node_t *actual, int interval[], double capacity){
	if(actual == NULL) {
		spdlog::debug("Don't existis any node that holds the specific interval [{};{}]", interval[0], interval[1]);
		return actual;
	}

	if(interval[0] < actual->interval[0]) {
		actual->left = remove_iteration(actual->left, interval, capacity);
	} else if(interval[0] > actual->interval[0]) {
		actual->right = remove_iteration(actual->right, interval, capacity);
	} else{ // the nodes has the same low interval
		if(interval[1] < actual->interval[1]) {
			actual->left = remove_iteration(actual->left, interval, capacity);
		} else if(interval[1] > actual->interval[1]) {
			actual->right = remove_iteration(actual->right, interval, capacity);
		} else{ // inside the node that need to be freed.
			if(actual->capacity < capacity) {
				SPDLOG_ERROR("The capacity to remove is higher than the current node has! To remove {}, has {}",actual->capacity, capacity);
				printf("The difference %.16f\n", actual->capacity - capacity);
				exit(0);
			}
			actual->capacity -= capacity;
			if(actual->capacity > 0) // the node has resources in use, so don't free it. Just return it
				return actual;

			//now,  the node need to be deleted, 'cause his capacity is equals 0, so check the possibilities, no child, 1 child and 2 children.
			if( (actual->left == NULL) || (actual->right == NULL)) {// the node has one or none child
				node_t *temp = actual->left ? actual->left : actual->right;
				if(temp == NULL) { // no child
					temp = actual;
					actual = NULL;
				}else{ // has one child
					copy_node(actual, temp);
					actual->left = NULL;
					actual->right= NULL;
				}
				free(temp);
				temp = NULL;
			} else{ //the node has two children
				node_t *temp = getMaxSubtree(actual->left);

				copy_node(actual, temp);

				int temp_interval[2] = {temp->interval[0], temp->interval[1]};
				actual->left = remove_iteration(actual->left, temp_interval, temp->capacity);
			}
		}
	}

	if(actual == NULL)
		return actual;

	actual->height = getMax(getHeight(actual->left),getHeight(actual->right)) + 1;

	actual->size = getSize(actual->left)+getSize(actual->right);

	//check the balancing factor
	int bf = getBF(actual);

	if(bf > 1 && getBF(actual->right) >= 0) { // single left
		return single_rotate_left(actual);
	}
	if(bf < -1 && getBF(actual->left) < 0) { // single right
		return single_rotate_right(actual);
	}
	if(bf > 1 && getBF(actual->right) < 0) { // double left
		actual->right = single_rotate_right(actual->right);
		return single_rotate_left(actual);
	}
	if(bf < -1 && getBF(actual->left) >= 0) { // double right
		actual->left = single_rotate_left(actual->left);
		return single_rotate_right(actual);
	}
	return actual;
}


inline void shrink_tree_to_interval(Interval_Tree *dst, node_t *src, int low, int high){
	if(src == NULL) return;

	if(hasOverlapShrink(src->interval[0], src->interval[1], low, high)) {
		int t_low = src->interval[0]  < low ? low : src->interval[0];
		int t_high = src->interval[1]  > high ? high :  src->interval[1];
		dst->insert(t_low, t_high, src->capacity);
	}

	shrink_tree_to_interval(dst, src->left, low, high);
	shrink_tree_to_interval(dst, src->right, low, high);
}

inline node_t *merge_nodes_iteration(node_t *actual){
	if(actual == NULL) return actual;

	node_t *temp = overlapSearch(actual, tree->root);
	if(temp == NULL) {
		if(actual->left != NULL) {
			temp = merge_nodes_iteration(actual->left);
		}
		if(actual->right != NULL) {
			temp = merge_nodes_iteration(actual->right);
		}
		return actual;
	} else {
		int low_actual, low_temp, low_new, high_actual, high_temp, high_new, old_low_temp, old_high_temp, old_low_actual, old_high_actual;

		double c_actual, c_temp, c_new, c_old_temp, c_old_actual;

		old_low_temp = temp->interval[0];
		old_low_actual = actual->interval[0];
		old_high_temp = temp->interval[1];
		old_high_actual = actual->interval[1];
		c_old_temp = temp->capacity;
		c_old_actual = actual->capacity;

		if (actual->interval[0] != temp->interval[0] && actual->interval[1] != temp->interval[1]) { // the merge will create a new node
			// Updating the nodes
			low_temp = getMin(actual->interval[0], temp->interval[0]);
			low_actual = getMax(actual->interval[0], temp->interval[0]);
			low_new = getMin(actual->interval[1], temp->interval[1]);

			high_temp = low_actual;
			high_actual = low_new;
			high_new = getMax(actual->interval[1], temp->interval[1]);

			if(actual->interval[0] > temp->interval[0]) {
				c_temp = temp->capacity;
			} else {
				c_temp = actual->capacity;
			}

			c_actual = actual->capacity + temp->capacity;

			if(actual->interval[1] > temp->interval[1]) {
				c_new = actual->capacity;
			} else {
				c_new = temp->capacity;
			}


			//remove the current nodes
			remove(old_low_actual, old_high_actual, c_old_actual);
			remove(old_low_temp, old_high_temp, c_old_temp);

			// printf("MERGE NODES - OLD TEMP [%d,%d] -> %lf\n",old_low_temp, old_high_temp, c_old_temp);
			// printf("MERGE NODES - OLD ACTUAL [%d,%d] -> %lf\n",old_low_actual, old_high_actual, c_old_actual);

			// printf("MERGE NODES - NEW TEMP [%d,%d] -> %lf\n",low_temp, high_temp, c_temp);
			// printf("MERGE NODES - NEW ACTUAL [%d,%d] -> %lf\n",low_actual, high_actual, c_actual);
			// printf("MERGE NODES - NEW NEW [%d,%d] -> %lf\n",low_new, high_new, c_new);

			//insert all the tree nodes
			insert(low_temp, high_temp, c_temp);
			insert(low_actual, high_actual, c_actual);
			insert(low_new, high_new, c_new);
		} else {
			if(actual->interval[0] == temp->interval[0]) { // only update the current nodes
				low_temp = temp->interval[0];
				low_actual = getMin(actual->interval[1], temp->interval[1]);

				high_temp = low_actual;
				high_actual = getMax(actual->interval[1], temp->interval[1]);

				if(actual->interval[1] > temp->interval[1]) {
					c_actual = actual->capacity;
				} else {
					c_actual = temp->capacity;
				}
				c_temp = temp->capacity + actual->capacity;
			} else {
				low_temp = getMin(actual->interval[0], temp->interval[0]);
				low_actual = getMax(actual->interval[0], temp->interval[0]);

				high_temp = low_actual;
				high_actual = actual->interval[1];

				if(actual->interval[0] < temp->interval[0]) {
					c_temp = actual->capacity;
				} else {
					c_temp = temp->capacity;
				}
				c_actual = actual->capacity + temp->capacity;
			}
			//remove the current nodes
			// printf("MERGE NODES - OLD TEMP [%d,%d] -> %lf\n",old_low_temp, old_high_temp, c_old_temp);
			// printf("MERGE NODES - OLD ACTUAL [%d,%d] -> %lf\n",old_low_actual, old_high_actual, c_old_actual);

			// printf("MERGE NODES - NEW TEMP [%d,%d] -> %lf\n",low_temp, high_temp, c_temp);
			// printf("MERGE NODES - NEW ACTUAL [%d,%d] -> %lf\n",low_actual, high_actual, c_actual);


			remove(old_low_temp, old_high_temp, c_old_temp);
			remove(old_low_actual, old_high_actual, c_old_actual);

			//insert all the tree nodes
			insert(low_temp, high_temp, c_temp);
			insert(low_actual, high_actual, c_actual);

		}
		return merge_nodes_iteration(tree->root);
	}
}

inline void merge_nodes(){
	merge_nodes_iteration(tree->root);
}

public:
Interval_Tree(double capacity = 0){
	if((tree = (interval_tree_t*)calloc(1, sizeof(interval_tree_t)))==NULL) {
		SPDLOG_ERROR("Error in malloc");
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

void setCapacity(double capacity){
	tree->capacity = capacity;
}

void addCapacity(double capacity){
	tree->capacity = capacity;
}

void subtractCapacity(double capacity){
	tree->capacity = capacity;
}

double getCapacity(){
	return tree->capacity;
}

void show(){
	spdlog::debug("The Bottleneck Interval Tree ");
	if(tree->root==NULL) {
		spdlog::debug("don't has any elements.");
	}else{
		spdlog::debug("has {} elements.", tree->root->size+1);
		show_bf(tree->root, 0);
	}
}

void insert(int low, int high, double capacity){
	if((capacity > tree->capacity) && ((capacity - tree->capacity) > 0.000001)) {
		SPDLOG_ERROR("The interval can't consume more capacity than the tree capacity ({} > {}) in time [{},{}]\n", capacity, tree->capacity, low, high);
		printf("The difference is %.16f > %.16f\n", capacity, tree->capacity);
		// TODO ADICIONOU O HASOVERLAPSHRINK, POIS QUANDO TEM QUE VERIFICAR SE OS OVERLAPS ESTAO CERTOS. QUANDO FAZ O SHRINK, O OVERLAP DEVE CONSIDERAR OS INTERVALOS IGUAIS, QUANDO FAZ O MERGE NAO DEVE?
		show();
		// exit(0);
	}
	// double maxValueAvailable = getMinValueAvailable(low, high);
	// if( maxValueAvailable >= capacity)
	tree->root = insert_iteration(tree->root, create_node(low, high, capacity));
	// else {
	// 	SPDLOG_ERROR("TENTANDO INSERIR MAIS RECURSOS DO QUE DISPONIVEL NA ARVORE {} < {} | Tree capacity {}", maxValueAvailable, capacity, tree->capacity);
	// 	printf("Difference of %.10f\n",maxValueAvailable - capacity);
	// 	exit(0);
	// }
}

void remove(int low, int high, double capacity){
	int interval[2] = {low, high};
	tree->root = remove_iteration(tree->root, interval, capacity);
}

interval_t* getInterval(int p_key, int s_key){// retorna um vetor com todos os intervalos e seus consumos de recursos
	//At first, it is needed to transform the tree to make the tree contain only nodes that overlap the interval [p_key, s_key].
	// After the new tree is built, merge the nodes that contain some type of overlap.
	Interval_Tree aux_tree(tree->capacity);

	// printf("SHRINK\n");
	shrink_tree_to_interval(&aux_tree, tree->root, p_key, s_key);

	if(aux_tree.empty()) return NULL;

	aux_tree.show();
	//Merge all the nodes that are in the specified interval
	// printf("MERGE\n");
	aux_tree.merge_nodes();
	// printf("MERGED\n");
	//Finally, when the merge of all nodes were done, convert the tree into a sorted array. To do that, get the left subtree first, then the parent node and then the right subtree.
	// printf("RESULTS\n");
	interval_t *result = (interval_t*) calloc (1, sizeof(interval_t));

	result->size = aux_tree.tree->root->size+1;
	result->nodes = (node_interval_t*) calloc (result->size, sizeof(node_interval_t));

	// printf("COPY TO ARRAY\n");
	aux_tree.copyToArray(result);

	// printf("COPIED\n");
	aux_tree.show();
	// Return the array.
	return result;
}

inline float getMinValueAvailable(int p_key, int s_key){// retorna um double com a menor capacidade do intervalo
	// printf("GET MIN INTERVAL [%d,%d]\n", p_key, s_key);
	interval_t *interval = getInterval(p_key, s_key);

	// printf("GET INTERVAL FUNCIONOU\n");
	double capacity = 0;

	// printf("VENDO SE O INTERVALO E VAZIO\n");
	if(interval==NULL) return static_cast<float>(tree->capacity);

	// printf("VENDO A MAIOR CAPACIDADE\n");
	for(int i=0; i < interval->size; i++) {
		capacity = capacity > interval->nodes[i].capacity ? capacity : interval->nodes[i].capacity;
	}
	// printf("LIMPANDO O INTERVALO\n");
	interval->clear();
	free(interval);
	interval = NULL;

	// printf("RETORNANDO A CAPACIDADE DISPONIVEL %lf\nTREE: %lf, Capacity %lf\n", tree->capacity - capacity, tree->capacity, capacity);
	return static_cast<float>(tree->capacity - capacity);
}

inline bool empty(){
	return tree->root == NULL ? true : false;
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
		SPDLOG_ERROR("The tree can't has capacity less than 0");
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

Interval_Tree& operator+= (double rhs){
	this->tree->capacity += rhs;
	return *this;
}

Interval_Tree& operator-= (double rhs){
	if(rhs > this->tree->capacity) {
		SPDLOG_ERROR("Operator -= The tree can't has capacity less than 0\n");
		return *this;
	}
	this->tree->capacity -= rhs;
	return *this;
}

};

}
#endif
