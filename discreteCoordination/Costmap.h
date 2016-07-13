/*
 * CostMap.h
 *
 *  Created on: Jul 12, 2016
 *      Author: andy
 */

#ifndef COSTMAP_H_
#define COSTMAP_H_

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>

using namespace std;

class Costmap {
public:
	Costmap();
	virtual ~Costmap();

	int nCols, nRows;
	int obsFree, infFree, unknown, obsWall, infWall, inflatedWall;
	// 1 = free space // 2 = inferred free space
	// 101 = unknown
	// 201 = wall // 202 = inferred wall // 203 = inflated wall
	vector<vector<int> > cells;

	vector<vector<float> > euclidDist; // array of distances
	float getEuclidDist(int x0, int y0, int x1, int y1);
	void getDistGraph();

	float aStarDist(vector<int> sLoc, vector<int> gLoc);
	vector<vector<int> > aStarPath(vector<int> sLoc, vector<int> gLoc);

	// TODO only update portion that needs it
	//void updateCostmap(vector<vector<int> > cells, vector<int> value);


};

#endif /* COSTMAP_H_ */
