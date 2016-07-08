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
#include "Agent.h"

using namespace cv;
using namespace std;

class Graph {
public:
	Graph();
	void createGraph(World &gMap, float obsThresh, float comThresh, int gSpace);
	virtual ~Graph();

	void initializeCostMap(); // Agents map for exploring

	vector<vector<int> > dStarPath(int gIndex,        Agent &bot, World &gMap);
	float dStarDist(int gIndex, Agent bot, World &gMap);
	vector<vector<int> > aStarPath(vector<int> sLoc, vector<int> gLoc, World &gMap);
	float aStarDist(vector<int> sLoc, vector<int> gLoc, World &gMap);


	Mat makeGlobalInferenceMat(World &gMap);
	Mat makeGeometricInferenceMatForMiniMap();
	Mat makeStructuralInferenceMatForMiniMap();
	Mat makeVisualInferenceMatForMiniMap();
	Mat makeNaiveMatForMiniMap();
	int wallInflationDistance;
	void inflateWalls(Mat &costMat);

	void observe(vector<int> cLoc, World &gMap); // what can the Agent see from cLoc
	void findFrontiers(); // search Graph and find Frontier
	void getFrontierCosts(World &gMap, vector<Agent> &bot); // find the dist to each Frontier
	void findNearestFrontier(Agent &bot, World &gMap); // find the closest Frontier to me
	void shareMap(vector<vector<float> >& in); // share maps with another Agent
	void clusterFrontiers(World &gMap); // cluster Frontiers into groups
	vector< vector<int> > kMeansClusteringTravel(vector<int> openFrnt, int numClusters, World &gMap); // cluster using travel dist
	vector< vector<int> > kMeansClusteringEuclid(vector<int> openFrnt, int numClusters, World &gMap); // cluster using euclidian

	int nCols;
	int nRows;

	void centralMarket(World &gMap, vector<Agent> &bot);

	Mat createMiniMapImg();
	Mat inferredMiniMap;
	vector<vector<int> > hullPts;
	Mat createMiniMapInferImg();

	void makeInferenceAndSetFrontierRewards(); // create outer hull and divide into contours
	void visualInference(); // perform visual inference on each contour
	void valueFrontiers(); // take the contours from inference and get the value of each
	void addExternalContours(vector<Point> outerHull, vector<vector<Point> > &externalContours, vector<int> frontierExit);
	vector<Frontier> frontiers; // graphs frontiers of item

	void getOuterHull(Mat &inferCalc, Mat &outerHullDrawing, vector<Point> &outerHull);
	vector<vector<Point> > getMinimalInferenceContours(Mat inferenceSpace);
	vector<float> getInferenceContourRewards(vector<int> frontierExits, vector<vector<Point> > contours);
	void setFrontierRewards(vector<float> rewards, vector<vector<Point> > inferenceContours);
	void displayInferenceMat(Mat &outerHullDrawing, Mat &obstacleHull, vector<Point> &outerHull, vector<int> frontierExits);

	void structuralBagOfWordsInference(vector<vector<Point> > &contours, Mat &obstaclesAndHull);
	void visualBagOfWordsInference(vector<vector<Point> > &contours, Mat &obstaclesAndHull);

	void clusteringObstacles();

	void compareFrontierValues(); // compare the calculated value against the global value
	void getFrontierOrientation();
	void getFrontierProjections();
	vector<int> getFrontierExits(vector<Point> &outerHull);

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
	float minMatchStrength;

	Mat obstacleMat;
	Mat freeMat;
	Mat FrontierMat;
	Mat unknownMat;
	Mat inferenceMat;

	vector<Point> getImagePoints(Mat &image); // convert mat to vector of points

	int getMinIndex(vector<float> value);
	int getMaxIndex(vector<float> value);

	void showCostMapPlot(int index);
	void buildCostMapPlot(World &gMap);
	void addAgentToPlot(World &gMap, Agent &bot);
	Mat image, tempImage;

	bool frntsExist; // do Frontiers exist?

	vector<vector<float> > costMap; // what the Agent uses to navigate
	// 1 = free space
	// 2 = inferred free space
	// 101 = unknown
	// 201 = wall
	// 202 = inferred wall
	// 203 = inflated wall

	float obsThresh;
	float comThresh;

	int gSpace;

	vector<int> cLoc;

	vector<vector<int> > frontiersList; // list of all Frontiers
	vector<vector<int> > frntCentroidList; // list of all Frontier Centroids
};

#endif /* SRC_Graph_H_ */
