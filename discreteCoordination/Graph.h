/*
 * Graph.h
 *
 *  Created on: Mar 2, 2016
 *      Author: andy
 */

#ifndef SRC_Graph_H_
#define SRC_Graph_H_

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>

#include "World.h"
#include "Frontier.h"
#include "Contour.h"
#include "Inference.h"

using namespace cv;
using namespace std;

class Graph {
public:
	Graph();
	void createGraph(World &gMap, float obsThresh, float comThresh);
	virtual ~Graph();

	// useful things
	int cNode; // my location
	vector<vector<int> > aStarPath(int sNode, int gNode);
	float aStarDist(int sNode, int gNode);

	// this holds the graph
	vector<Node> nodes;

	void initializeCostMap(); // Agents map for exploring

	// Frontiers
	void findFrontiers(); // search Graph and find Frontier
	vector<Frontier> frontiers; // graphs frontiers of item

	vector< vector<int> > kMeansClusteringTravel(vector<int> openFrnt, int numClusters, World &gMap); // cluster using travel dist
	vector< vector<int> > kMeansClusteringEuclid(vector<int> openFrnt, int numClusters, World &gMap); // cluster using euclidian

	int nCols;
	int nRows;

	void centralMarket();

	Mat createMiniMapImg();

	void compareFrontierValues(); // compare the calculated value against the global value

	Mat getObstaclesImage();
	Mat getFrontiersImage();
	Mat getFreeSpaceImage();
	Mat getInferenceImage();


	Mat obstacleMat;
	Mat freeMat;
	Mat FrontierMat;
	Mat unknownMat;
	Mat inferenceMat;

	vector<Point> getImagePoints(Mat &image); // convert mat to vector of points

	int getMinIndex(vector<float> value);
	int getMaxIndex(vector<float> value);

	void showCostMapPlot(int index);
	void buildCostMapPlot();
	void addAgentToPlot(Scalar color, vector<vector<int> > myPath, vector<int> cLoc);
	Mat image, tempImage;

	bool frntsExist; // do Frontiers exist?

	float obsThresh;
	float comThresh;
	vector<vector<int> > viewPerim;


	vector<vector<int> > poseSet;
	void findPoseSet(Mat &costMat);
	void simulateObservation(vector<int> pose, Mat &resultingView, Mat &costMat);
	int matReward(Mat &in);

};

#endif /* SRC_Graph_H_ */
