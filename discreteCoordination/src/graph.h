/*
 * graph.h
 *
 *  Created on: Mar 2, 2016
 *      Author: andy
 */

#ifndef SRC_GRAPH_H_
#define SRC_GRAPH_H_

#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>

#include "world.h"
#include "frontier.h"
#include "agent.h"

using namespace cv;
using namespace std;

class graph {
public:
	graph();
	void createGraph(world &gMap, float obsThresh, float comThresh, int gSpace);
	virtual ~graph();

	void initializeCostMap(); // agents map for exploring

	vector<vector<int> > dStarPath(frontier &frnt, agent &bot, world &gMap);
	float dStarDist(frontier &frnt, agent bot, world &gMap);
	vector<vector<int> > aStarPath(vector<int> sLoc, vector<int> gLoc, world &gMap);
	float aStarDist(vector<int> sLoc, vector<int> gLoc, world &gMap);

	void observe(vector<int> cLoc, world &gMap); // what can the agent see from cLoc
	void findFrontiers(); // search graph and find frontier
	void frontierCosts(world &gMap, vector<frontier> &frnt, vector<agent> &bot); // find the dist to each frontier
	void findNearestFrontier(agent &bot, vector<frontier> &frnt, world &gMap); // find the closest frontier to me
	void shareMap(vector<vector<float> >& in); // share maps with another agent
	void clusterFrontiers(world &gMap, vector<frontier> &frnt); // cluster frontiers into groups
	vector< vector<int> > kMeansClusteringTravel(vector<int> openFrnt, int numClusters, world &gMap); // cluster using travel dist
	vector< vector<int> > kMeansClusteringEuclid(vector<int> openFrnt, int numClusters, world &gMap); // cluster using euclidian

	int nCols;
	int nRows;

	void centralMarket(world &gMap, vector<frontier> &frnt, vector<agent> &bot);

	Mat createMiniMapImg();
	Mat inferredMiniMap;
	vector<vector<int> > hullPts;
	Mat createMiniMapInferImg();

	void makeInference(vector<frontier>  &frnt);
	void getFrontierOrientation(vector<frontier> & frnt);
	void getFrontierProjection();
	Mat getObstaclesImage();
	Mat getFrontiersImage();
	Mat getFreeSpaceImage();
	Mat getInferenceImage();
	void extractInferenceContour();
	void getLengthHistogram(vector<float> length, float meanLength, vector<int> &histogram, vector<float> &sequence);
	vector<vector<int> > masterHistogramList;
	vector<vector<float> > masterSequenceList;
	vector<vector<Point> > masterPointList;
	vector<String> masterNameList;
	vector<Point> masterCenterList;
	vector<float> masterMeanLengthList;

	Mat obstacleMat;
	Mat freeMat;
	Mat frontierMat;
	Mat unknownMat;
	Mat inferenceMat;

	vector<Point> getImagePoints(Mat &image); // convert mat to vector of points

	int getMinIndex(vector<float> value);
	int getMaxIndex(vector<float> value);

	void showCostMapPlot(int index);
	void buildCostMapPlot(world &gMap, vector<frontier> &frnt);
	void addAgentToPlot(world &gMap, vector<int> cLoc, int myColor[3]);
	Mat image, tempImage;

	bool frntsExist; // do frontiers exist?

	vector<vector<float> > costMap; // what the agent uses to navigate
	// 0 = free space
	// 50 = frontier
	// 100 = unknown
	// Infinity = obstacle

	float obsThresh;
	float comThresh;

	int gSpace;

	vector<int> cLoc;

	vector<vector<int> > frntList; // list of all frontiers
	vector<vector<int> > frntCentroidList; // list of all frontier Centroids
};

#endif /* SRC_GRAPH_H_ */
