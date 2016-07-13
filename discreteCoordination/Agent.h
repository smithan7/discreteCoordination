/*
 * Agent.h
 *
 *  Created on: Jun 8, 2016
 *      Author: andy
 */
/*
 * Agent.h
 *
 *  Created on: Mar 2, 2016
 *      Author: andy
 */

#ifndef SRC_Agent_H_
#define SRC_Agent_H_

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

#include "Graph.h"
#include "World.h"
#include "Costmap.h"

using namespace std;

class Agent{
public:

	// agent stuff
	Agent();
	void buildAgent(vector<int> sLoc, int myIndex, World &gMap, float comThresh, float obsThresh);
	~Agent();
	void shareCostmap(Costmap &A, Costmap &B);
	void shareGoals(vector<int> inG, int index);
	int myIndex;
	int myColor[3];
	float comThresh;
	float obsThresh;

	// graph class stuff
	Graph graph;
	GraphCoordination graphCoordination;
	GraphPlanning graphPlanning;
	int cNode;
	int gNode;
	vector<int> nodePath;

	// costmap class stuff
	Costmap costmap;
	CostmapCoordination costmapCoordination;
	CostmapPlanning costmapPlanning;
	vector<int> cLoc; // for map
	vector<int> gLoc; // for map
	vector<vector<int> > myPath;

	// inference stuff
	Inference inference;


	int marketNodeSelect(World &gMap);
	void greedyFrontiers();
};

#endif /* SRC_Agent_H_ */
