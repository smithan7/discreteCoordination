/*
 * Frontier.cpp
 *
 *  Created on: Apr 9, 2016
 *      Author: andy
 */

#include "Frontier.h"

Frontier::Frontier(vector<vector<int> > q) {
	this->projection.push_back(0);
	this->projection.push_back(0);
	this->projectionDistance = 5;

	this->centroid.push_back(-1);
	this->centroid.push_back(-1);

	this->orient.push_back(-1);
	this->orient.push_back(-1);

	this->reward = 0;
	this->cost = INFINITY;
	this->area = 0;

	this->editFlag = true;

	this->members = q;
}

Frontier::~Frontier() {

}

void Frontier::getFrontierProjection(){
	this->projection[0] = this->centroid[0] + round(this->projectionDistance*this->orient[0]);
	this->projection[1] = this->centroid[1] + round(this->projectionDistance*this->orient[1]);
}

void Frontier::getFrontierOrientation(const vector<vector<int> > &costMap){
	vector<float> temp;
	temp.push_back(0);
	temp.push_back(0);
	this->orient = temp;
	float count = 0;
	// check each member of each cluster
	if(this->members.size() > 1){
		for(int j=0; j<(int)this->members.size(); j++){
			// check 4Nbr for being unobserved
			int xP = this->members[j][0];
			int yP = this->members[j][1];
			if(costMap[xP+1][yP] == 100){
				this->orient[0] += 1;
				count++;
			}
			if(costMap[xP-1][yP] == 100){
				this->orient[0] -= 1;
				count++;
			}
			if(costMap[xP][yP+1] == 100){
				this->orient[1] += 1;
				count++;
			}
			if(costMap[xP][yP-1] == 100){
				this->orient[1] -= 1;
				count++;
			}
		}
		if(count > 0){
			this->orient[0] /= count;
			this->orient[1] /= count;
		}
	}
}
