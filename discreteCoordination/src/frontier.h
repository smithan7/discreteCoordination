/*
 * frontier.h
 *
 *  Created on: Apr 9, 2016
 *      Author: andy
 */

#ifndef FRONTIER_H_
#define FRONTIER_H_

#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>

using namespace std;

class frontier {
public:
	frontier(int nRows, int nCols);
	virtual ~frontier();

	vector<float> orient; // unit vector descirbing orientation
	vector<int> centroid; // [x/y]
	vector<int> projection; // [x/y]
	int area; // area behind this frontier
	float reward; // reward for this frontier

	vector<vector<int> > obstacles;
	vector<vector<vector<int> > > obsClusters;

	vector<vector<int> > members; // [list][x/y]
	bool editFlag;

	int nRows;
	int nCols;

	vector<vector<int> > cSet; // 1 means in closed set, 0 means not
	vector<vector<int> > oSet; // 1 means in open set, 0 means not
	vector<vector<float> > gScore; // known cost from initial node to n
	vector<vector<float> > fScore; // gScore + heuristic score (dist to goal + imposed cost)
	vector<vector<vector<int> > > cameFrom; // each square has a vector of the location it came from
	void getFrontierProjection();



};

#endif /* FRONTIER_H_ */
