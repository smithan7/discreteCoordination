/*
 * miniGraph.h
 *
 *  Created on: Mar 14, 2016
 *      Author: andy
 */

#ifndef miniGraph_H_
#define miniGraph_H_

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>

#include "frontier.h"

using namespace cv;
using namespace std;

class miniGraph {
public:
	miniGraph();
	virtual ~miniGraph();

	//void buildTree();

	void createMiniGraph(Mat &bw, int minX, int minY, int maxX, int maxY); // create miniGraph from explore image, change to do from explore graph, on my nodes not pixels
	void findPointOfInterestNodes(); // use edge detector to find poi nodes for minigraph, result is too sparse
	void breadthFirstSearchAssembleMiniGraph(); // search through the graph looking for connections using Dijstras
	int grafSpacing;
	void findCityBlockDistanceNodes(vector<vector<int> > miniGraphNodes); // use city block distance to find nodes
	void cityBlockDistanceNodeConnections(); // use city block distance to get dist graph
	void getUnobservedMat(Mat &inputMat);
	void getInferenceMat(Mat &inputMat);



	// inference tools
	void extractInferenceContour();
	void growObstacles();
	Mat breadthFirstSearchFindRoom(Mat &src, vector<int> pt); // search along walls, need to find way to exclude dominated points (do by searching next to wall in observed space!)
	void growFrontiers(vector<frontier> frnt); // grow frontiers towards eachother
	void invertImageAroundPt(Mat &src, Mat &dst, vector<int> cLoc);

	// observation tools
	bool lineTraversabilityCheck(Mat &tSpace, vector<int> sPt, vector<int> fPt, int fValue);
	bool bisectionCheck(vector<int> a, vector<int> b); // for checking visibility
	bool bresenhamLineCheck(vector<int> cLoc, vector<int> cPt); // for checking visibility
	void observe(vector<int> cLoc, Mat &viewMat);
	vector<vector<int> > corners;
	void cornerFinder(Mat &inputImage);
	Mat obstacleMat;
	Mat freeMat;
	Mat frontierMat;
	Mat unknownMat;
	Mat inferenceMat;
	vector<vector<int> > viewPerim; // list of perimeter nodes on circle around the UAV that represent all points viewable, use for observation check

	float matReward(Mat &in); // calc how much of the Mat has been observed.

	void floodForDist(); // create the distgraph
	void thinning(const Mat& src, Mat& dst, int minX, int minY, int maxX, int maxY); // used to create miniGraph
	void thinningIteration(Mat& img, int iter); // used to create miniGraph

	void drawCoordMap(vector<int> cLoc); // draw the coordination map, locations and

	void findMyNode(vector<int> cLoc);
	int findNearestNode(vector<int> in); // find the node closest to point
	//void frontierCosts(); // distance to each frontier cluster
	void importFrontiers(vector<vector<int> > frntList); // bring frontiers in from graph
	void getNodeCosts(int cNode); // distance to travel to each node
	void getNodeRewards(); // reward for each node, currently number of frontiers around it
	void getNodeValues(); // value = reward - cost

	int getMaxIndex(vector<float> value); //
	vector<int> greedyNodeSelect(); // go to nearest node with frontiers
	vector<int> marketNodeSelect(); // go to the highest value node

	vector<int> aStar(int strt, int gl);
	float aStarDist(int sLoc, int eLoc); // distance along travel graph from sLoc to eLoc
	float cityBlockDist(vector<int> a, vector<int> b); // fast dist calc
	float euclidianDist(vector<int> a, vector<int> b); // accurate dist calc

	void importUAVLocations(vector<vector<int> > cLocList);
	vector<vector<int> > cLocListMap;
	vector<int> cLocListNodes;
	/*
	void myThinning(vector<graphNode> &space);
	void createMiniGraph2(vector<graphNode> &graf);
	bool checkNodeForThinning(int x, vector<graphNode> &graf); // should this node be pruned?
	bool checkPerimDerivative(int x, vector<graphNode> &graf);
	bool checkSumNbrs(int x, vector<graphNode> &graf);
	bool checkNbrs(int x, vector<graphNode> &graf, int condit);



	vector<graphNode> miniSpace;
	*/
	Mat miniImage;
	int nmNodes; // number of nodes on graph
	vector<vector<int> > graf; // list of all nodes on the miniGraph
	vector<vector<float> > distGraph; // distances between each node on the travel graph
	float scaling; // how much is my image scaled down from graph

	vector<vector<int> > cLocList; // list of UAV locations

	vector<vector<int> > frontiers; // list of frontier centers
	vector<float> frntCost; // cost to travel to each frontier along the graph
	vector<vector<int> > nodeFrontiers; // frontiers at each node

	vector<float> nodeCost; // cost to travel to each node
	vector<float> nodeReward; // reward of each node
	vector<float> nodeValue; // value of each node

	int cNode; // which node am I located at

};

#endif /* miniGraph_H_ */
