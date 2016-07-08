/*
 * frontier.cpp
 *
 *  Created on: Apr 9, 2016
 *      Author: andy
 */

#include "frontier.h"

frontier::frontier(int nRows, int nCols) {
	// TODO Auto-generated constructor stub

	this->projection.push_back(-1);
	this->projection.push_back(-1);

	this->centroid.push_back(-1);
	this->centroid.push_back(-1);

	this->orient.push_back(-1);
	this->orient.push_back(-1);

	this->nRows = nRows;
	this->nCols = nCols;

	this->reward = 0;
	this->area = 0;

	this->editFlag = true;

	for(int i=0;i<nRows; i++){
		vector<int> tC;
		vector<int> tO;
		vector<vector<int> > tCF;
		vector<float> tG;
		vector<float> tF;
		for(int j=0; j<nCols; j++){
			tC.push_back(0);
			tO.push_back(0);
			vector<int> ttCF;
			ttCF.push_back(-1);
			ttCF.push_back(-1);

			tCF.push_back(ttCF);
			tG.push_back(INFINITY);
			tF.push_back(INFINITY);
		}
		this->cSet.push_back(tC);
		this->oSet.push_back(tC);
		this->cameFrom.push_back(tCF);
		this->gScore.push_back(tG); // init scores to inf
		this->fScore.push_back(tF); // init scores to inf
	}

}

frontier::~frontier() {
	// TODO Auto-generated destructor stub
}

void frontier::getFrontierProjection(){
	this->projection[0] = this->centroid[0] + round(2*this->orient[0]);
	this->projection[1] = this->centroid[1] + round(2*this->orient[1]);
}
