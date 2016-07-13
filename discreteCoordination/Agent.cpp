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

void Agent::shareMap(Costmap &A, Costmap &B){
	for(int i=0; i<A.nCols; i++){
		for(int j=0; j<A.nRows; j++){
			if(A.cells[i][j] == 101 && B.cells[i][j] != 101){ // I dont think its observed, they do
				A.cells[i][j] = B.cells[i][j];
			}
			else if(A.cells[i][j] != 101 && B.cells[i][j] == 101){
				B.cells[i][j] = A.cells[i][j];
			}
		}
	}
}

void Agent::shareGoals(vector<int> inG, int inI){
	this->goalList[inI] = inG;
}

Agent::~Agent() {

}

