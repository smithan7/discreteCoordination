/*
 * Contour.h
 *
 *  Created on: Jul 10, 2016
 *      Author: andy
 */

#ifndef CONTOUR_H_
#define CONTOUR_H_


#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>

using namespace std;

class Contour {
public:
	Contour();
	virtual ~Contour();

	int value;
	int area;

	vector<vector<int> > members; // [k][x,y]
	vector<vector<int> > perimeter; // [k][x,y,nbr value];
};

#endif /* CONTOUR_H_ */
