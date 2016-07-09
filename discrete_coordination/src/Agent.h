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

#include "MiniGraph.h"
#include "World.h"

using namespace std;

class Agent{
public:
	//graph myMap;
	//miniGraph myMiniMap;

	int myColor[3];
	vector<int> cLoc; // for map
	vector<int> gLoc; // for map
	int gIndex; // index of frontier I am after

	int cNode; // for miniMap
	int gNode; // for miniMap

	float comThresh;
	float obsThresh;

	int myIndex;
	vector<vector<int> > myPath;
	vector<vector<int> > goalList;
	vector<float> fCost;
	float gLocValue;
	float fRadius;
	vector<float> minPubCost;

	Agent();
	void buildAgent(vector<int> sLoc, int myIndex, World &gMap, float comThresh, float obsThresh);
	void aStarPathPlanning(World &gMap);
	float getFrontierCost(int fIndex, World &gMap);
	void shareGoals(vector<int> inG, int index);
	int marketNodeSelect(World &gMap);
	void greedyFrontiers();
	virtual ~Agent();
};

#endif /* SRC_Agent_H_ */
