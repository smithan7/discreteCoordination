/*
 * CostmapCoordination.h
 *
 *  Created on: Jul 12, 2016
 *      Author: andy
 */

#ifndef COSTMAPCOORDINATION_H_
#define COSTMAPCOORDINATION_H_

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>

#include "World.h"
#include "Frontier.h"
#include "Costmap.h"

class CostmapCoordination {
public:
	CostmapCoordination(vector<vector<float> > euclidDist);
	virtual ~CostmapCoordination();

	// useful stuff
	float getEuclidDist(int x0, int y0, int x1, int y1);
	vector<vector<float> > euclidDist;
	float aStarDist(vector<int> sLoc, vector<int> gLoc, vector<vector<int> > &costMap);

	// main coordination
	vector<int> getGoal(vector<int> cLoc, Costmap &costmap);

	// Frontiers
	vector<Frontier> frontiers; // graphs frontiers of item
	vector<vector<int> > findFrontiers(Costmap &costmap); // search Graph and find Frontiers
	void clusterFrontiers(vector<vector<int> > frntList); // cluster Frontiers into groups

	// for frontier markets
	void getFrontierCosts(vector<int> cLoc,  Costmap &costmap); // find the dist to each Frontier
	void importFontierBids(vector<vector<float> > frontierBids); // [agent][value]
	vector<int> marketFrontiers(); // bid over the frontiers and choose my frontier

	vector< vector<int> > kMeansClusteringTravel(int numClusters, Costmap &costmap); // cluster using travel dist
	vector< vector<int> > kMeansClusteringEuclid(int numClusters, Costmap &costmap); // cluster using euclidian
	vector<vector<int> > centralMarket(Costmap &costmap, vector<vector<int> > cLoc);

	// greedy frontier planning
	vector<int> findNearestFrontier(); // find the closest Frontier to me

};

#endif /* COSTMAPCOORDINATION_H_ */
