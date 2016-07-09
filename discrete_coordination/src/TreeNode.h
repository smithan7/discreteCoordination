/*
 * TreeNode.h
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

using namespace std;

class TreeNode{
public:
	TreeNode(int state, vector<vector<float> > &distGraph, vector<Mat> &observations, vector<int> inPath, int depth, float pathLength);
	virtual ~TreeNode();

	vector<TreeNode> children;
	int myState; // link to physical graph representation
	float reward; // my current value
	int nPulls; // number of times I've been pulled
	vector<int> myPath;
	int myDepth;
	bool searchComplete;
	float pathLength;

	void getChildren(vector<vector<float> > &distGraph, vector<Mat> &observations);
	void getNodeReward(vector<Mat> &observations);
	// select children algorithms
	int UCBChildSelect();
	int eGreedyChildSelect(float epsilon);
	int greedyChildSelect();
	int simAnnealingChildSelect(float& temp, float cooling);
	int randChildSelect();


	void updateMyReward();
	void searchTree(vector<vector<float> > &distGraph, vector<Mat> &observations, float maxDist);
	void exploitTree(vector<int>& myPath, vector<float> &rewards);

	void deleteTree();
};

#endif /* TREENODE_H_ */
