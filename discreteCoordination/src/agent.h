/*
 * agent.h
 *
 *  Created on: Mar 2, 2016
 *      Author: andy
 */

#ifndef SRC_AGENT_H_
#define SRC_AGENT_H_

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

using namespace std;

#include "world.h"

class agent{
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

	agent();
	void buildAgent(vector<int> sLoc, int myIndex, world &gMap, float comThresh, float obsThresh);
	void aStarPathPlanning(world &gMap);
	float getFrontierCost(int fIndex, world &gMap);
	void shareGoals(vector<int> inG, int index);
	int marketNodeSelect(world &gMap);
	void greedyFrontiers();
	virtual ~agent();
};

#endif /* SRC_AGENT_H_ */
