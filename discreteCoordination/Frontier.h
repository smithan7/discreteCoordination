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
	Frontier(vector<vector<int> > members);
	virtual ~Frontier();

	vector<float> orient; // unit vector descirbing orientation
	vector<int> centroid; // [x/y]
	vector<int> projection; // [x/y]
	float projectionDistance;
	int area; // area behind this Frontier
	float reward; // reward for this Frontier
	float cost; // cost of travel to frontier for owning agent

	vector<vector<int> > obstacles;
	vector<vector<vector<int> > > obsClusters;

	vector<vector<int> > members; // [list][x/y]
	bool editFlag;

	void getFrontierProjection();
	void getFrontierOrientation(const vector<vector<int> > &costMap);



};

#endif /* Frontier_H_ */
