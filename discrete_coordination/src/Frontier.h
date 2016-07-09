/*
 * Frontier.h
 *
 *  Created on: Apr 9, 2016
 *      Author: andy
 */

#ifndef Frontier_H_
#define Frontier_H_

#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>

using namespace std;

class Frontier {
public:
	Frontier(int nRows, int nCols);
	virtual ~Frontier();

	vector<float> orient; // unit vector descirbing orientation
	vector<int> centroid; // [x/y]
	vector<int> projection; // [x/y]
	float projectionDistance;
	int area; // area behind this Frontier
	float reward; // reward for this Frontier

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

#endif /* Frontier_H_ */
