/*
 * Node.h
 *
 *  Created on: Jul 10, 2016
 *      Author: andy
 */

#ifndef NODE_H_
#define NODE_H_

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;


class Node {
public:
	Node(vector<int> loc);
	virtual ~Node();

	Mat observation;
	vector<int> loc;
	float reward;
	vector<int> nbrs;
	vector<float> transitions;
};

#endif /* NODE_H_ */
