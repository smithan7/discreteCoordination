/*
 * treeNode.h
 *
 *  Created on: May 18, 2016
 *      Author: andy
 */

#ifndef TREENODE_H_
#define TREENODE_H_

#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

#include "miniGraph.h"

using namespace std;

class treeNode{
public:
	treeNode(int state, miniGraph &miniMaster, vector<int> myPath, int depth);
	virtual ~treeNode();

	vector<treeNode> children;
	int myState; // link to physical graph representation
	float value; // my current value
	int nPulls; // number of times I've been pulled
	vector<int> myPath;
	int myDepth;
	bool searchComplete;

	void getChildren(miniGraph &miniMaster);
	void evaluateNode(miniGraph &miniMaster);
	// select children algorithms
	int UCBChildSelect();
	int eGreedyChildSelect(float epsilon);
	int greedyChildSelect();
	int simAnnealingChildSelect(float& temp, float cooling);
	int randChildSelect();


	void updateMyValue(float passedValue);
	void searchTree(miniGraph &miniMaster);
	void exploitTree(vector<int>& myPath);

	void deleteTree();
};

#endif /* TREENODE_H_ */
