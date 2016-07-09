/*
 * Agent.cpp
 *
 *  Created on: Mar 2, 2016
 *      Author: andy
 */

using namespace std;

#include "Agent.h"

Agent::Agent(){

}

void Agent::buildAgent(vector<int> sLoc, int myIndex, World &gMap, float obsThresh, float comThresh){
	this->obsThresh = obsThresh;
	this->comThresh = comThresh;
	//this->myMap.createGraph(gMap, obsThresh, comThresh, gMap.gSpace);
	this->gLocValue = 0;
	this->fRadius = 5*gMap.gSpace;

	this->cLoc.push_back(sLoc[0]);
	this->cLoc.push_back(sLoc[1]);

	this->gLoc.push_back(sLoc[0]);
	this->gLoc.push_back(sLoc[1]);

	this->gNode = -1;

	this->myIndex = myIndex;
	this->myColor[0] = 0;
	this->myColor[1] = 0;
	this->myColor[2] = 0;

	if(this->myIndex == 0){
		this->myColor[0] = 255;
	}
	else if(this->myIndex == 1){
		this->myColor[1] = 255;
	}
	else if(this->myIndex == 2){
		this->myColor[2] = 255;
	}
	else if(this->myIndex == 3){
		this->myColor[0] = 255;
		this->myColor[1] = 153;
		this->myColor[2] = 51;
	}
	else if(this->myIndex == 4){
		this->myColor[0] = 255;
		this->myColor[1] = 255;
		this->myColor[2] = 51;
	}
	else if(this->myIndex == 5){
		this->myColor[0] = 255;
		this->myColor[1] = 51;
		this->myColor[2] = 255;
	}
	else if(this->myIndex == 6){
		this->myColor[0] = 51;
		this->myColor[1] = 255;
		this->myColor[2] = 255;
	}
	else if(this->myIndex == 7){
		this->myColor[0] = 153;
		this->myColor[1] = 255;
		this->myColor[2] = 51;
	}
	else if(this->myIndex == 8){
		this->myColor[0] = 255;
		this->myColor[1] = 255;
		this->myColor[2] = 255;
	}
	else if(this->myIndex == 9){
		// white
	}
}

void Agent::shareGoals(vector<int> inG, int inI){
	this->goalList[inI] = inG;
}

int Agent::marketNodeSelect(World &gMap){
	//this->cNode = this->myMiniMap.findNearestNode(this->cLoc); // find node I am closest to
	//this->myMiniMap.getNodeCosts(this->cNode); // get node costs
	//this->myMiniMap.getNodeRewards(); // get node costs
	//this->myMiniMap.getNodeValues(); // get node costs
	//this->gNode = this->myMiniMap.getMaxIndex(this->myMiniMap.nodeValue);
	//this->gLoc[1] = this->myMiniMap.frontiers[this->myMiniMap.nodeFrontiers[this->gNode][0]][1];
	//this->gLoc[0] = this->myMiniMap.frontiers[this->myMiniMap.nodeFrontiers[this->gNode][0]][0];
	//cerr << "cNode: " << this->cNode << endl;
	//cerr << "gNode: " << this->gNode << endl;
	//this->myMiniMap.drawCoordMap(this->cLoc);

	/*
	this->myMap.frontierCosts(this->cLoc, gMap);
	int maxdex;
	float maxVal = 0;
	this->myMap.frntValue.erase(this->myMap.frntValue.begin(),this->myMap.frntValue.end());
	for(int i=0; i<this->myMap.frntCost.size(); i++){
		this->myMap.frntValue.push_back(this->myMap.frntReward[i] - this->myMap.frntCost[i]);
	}
	int fGoal = this->myMap.getMaxIndex(this->myMap.frntValue);

	this->gLoc = this->myMap.frntCentroid[fGoal];
	*/
}

void Agent::greedyFrontiers(){

}


float Agent::getFrontierCost(int fIndex, World &gMap){
//	float dist = this->myMap.aStarDist(this->cLoc, this->myMap.frntList[fIndex], gMap);
//	return dist;
}


void Agent::aStarPathPlanning(World &gMap){
//	this->myPath = this->myMap.aStarPath(this->cLoc, this->gLoc, gMap);
}

Agent::~Agent() {
	// TODO Auto-generated destructor stub
}

