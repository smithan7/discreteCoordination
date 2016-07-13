/*
 * Costmap.cpp
 *
 *  Created on: Jul 12, 2016
 *      Author: andy
 */




#include "Costmap.h"

Costmap::Costmap() {

}

Costmap::~Costmap() {

}

void Costmap::getDistGraph(){
	for(int i=0; i<this->nCols; i++){
		vector<float> d;
		for(int j=0; j<this->nRows; j++){
			d.push_back(sqrt(pow(i,2) + pow(j,2)));
		}
		this->euclidDist.push_back(d);
	}
}


float Costmap::getEuclidDist(int x0, int y0, int x1, int y1){
	int bx = abs(x0 - x1);
	int by = abs(y0 - y1);

	return(this->euclidDist[bx][by]);
}


vector<vector<int> > Costmap::aStarPath(vector<int> sLoc, vector<int> gLoc){
	float costMapPenalty;

	if(sLoc == gLoc){
		vector<vector<int> > totalPath;
		for(int i=0; i<4; i++){
			vector<int> t = sLoc;
			totalPath.push_back(t);
		}
		return totalPath;
	}
	vector<vector<int> > cSet; // 1 means in closed set, 0 means not
	vector<vector<int> > oSet; // 1 means in open set, 0 means not
	vector<vector<float> > gScore; // known cost from initial node to n
	vector<vector<float> > fScore; // gScore + heuristic score (dist to goal + imposed cost)
	vector<vector<vector<int> > > cameFrom; // each square has a vector of the location it came from
	for(int i=0;i<this->nRows; i++){
		vector<int> tC;
		vector<int> tO;
		vector<vector<int> > tCF;
		vector<float> tG;
		vector<float> tF;
		for(int j=0; j<this->nCols; j++){
			tC.push_back(0);
			tO.push_back(0);
			vector<int> ttCF;
			ttCF.push_back(-1);
			ttCF.push_back(-1);

			tCF.push_back(ttCF);
			tG.push_back(INFINITY);
			tF.push_back(INFINITY);
		}
		cSet.push_back(tC);
		oSet.push_back(tC);
		cameFrom.push_back(tCF);
		gScore.push_back(tG); // init scores to inf
		fScore.push_back(tF); // init scores to inf
	}

	oSet[sLoc[0]][sLoc[1]] = 1; // starting node has score 0
	gScore[sLoc[0]][sLoc[1]] = 0; // starting node in open set

	if(this->cells[sLoc[0]][sLoc[1]] < 102){
		costMapPenalty = 0;//this->costMap[sLoc[0]][sLoc[1]];
	}
	else{
		costMapPenalty = INFINITY;
	}
	fScore[sLoc[0]][sLoc[1]] = gScore[sLoc[0]][sLoc[1]] + this->getEuclidDist(sLoc[0],sLoc[1],gLoc[0],gLoc[1]) + costMapPenalty;

	int foo = 1;
	int finishFlag = 0;
	while(foo>0 && finishFlag == 0){
		/////////////////// this finds node with lowest fScore and makes current
		float min = INFINITY;
		vector<int> iMin;
		iMin.push_back(0);
		iMin.push_back(0);
		for(int i=0; i<this->nRows; i++){
			for(int j=0; j<this->nCols; j++){
				if(oSet[i][j] > 0 && fScore[i][j] < min){
					min = fScore[i][j];
					iMin[0] = i;
					iMin[1] = j;
				}
			}
		}
		vector<int> cLoc = iMin;
		/////////////////////// end finding current node
		if(cLoc == gLoc){ // if the current node equals goal, construct path
			finishFlag = 1;
			vector<vector<int> > totalPath;
			totalPath.push_back(gLoc);
			while(cLoc != sLoc){ // work backwards to start
				vector<int> temp;
				temp.push_back(cameFrom[cLoc[0]][cLoc[1]][0]); // work backwards
				temp.push_back(cameFrom[cLoc[0]][cLoc[1]][1]);
				cLoc = temp;
				totalPath.push_back(cLoc); // append path
			}
			reverse(totalPath.begin(),totalPath.end());
			return totalPath;
		} ///////////////////////////////// end construct path
		oSet[cLoc[0]][cLoc[1]] = 0;
		cSet[cLoc[0]][cLoc[1]] = 1;
		for(int nbrRow=cLoc[0]-1;nbrRow<cLoc[0]+2;nbrRow++){
			if(nbrRow >= 0 && nbrRow < this->nRows){
				for(int nbrCol=cLoc[1]-1; nbrCol<cLoc[1]+2; nbrCol++){
					if(nbrCol >= 0 && nbrCol < this->nRows){
						if(cSet[nbrRow][nbrCol] == 1){ // has it already been eval? in cSet
							continue;
						}
						float tGScore;
						tGScore = gScore[cLoc[0]][cLoc[1]] + this->getEuclidDist(sLoc[0],sLoc[1],nbrRow,nbrCol); // calc temporary gscore

						if(oSet[nbrRow][nbrCol] == 0){
							oSet[nbrRow][nbrCol] = 1;  // add nbr to open set
						}
						else if(tGScore >= gScore[nbrRow][nbrCol]){ // is temp gscore better than stored g score of nbr
							continue;
						}
						cameFrom[nbrRow][nbrCol][0] = cLoc[0];
						cameFrom[nbrRow][nbrCol][1] = cLoc[1];
						gScore[nbrRow][nbrCol] = tGScore;
						if(this->cells[nbrRow][nbrCol] < 102){
							costMapPenalty = 0;//this->costMap[nbrRow][nbrCol];
						}
						else{
							costMapPenalty = INFINITY;
						}
						fScore[nbrRow][nbrCol] = gScore[nbrRow][nbrCol] + this->getEuclidDist(gLoc[0],gLoc[1],nbrRow,nbrCol) + costMapPenalty;
					}
				}
			}
		}
		/////////////// end condition for while loop, check if oSet is empty
		foo = 0;
		for(int i=0; i<this->nRows; i++){
			for(int j=0; j<this->nCols; j++){
				foo+= oSet[i][j];
			}
		}
	}
	vector<vector<int> > totalPath;
	for(int i=0; i<4; i++){
		vector<int> t = sLoc;
		totalPath.push_back(t);
	}
	return totalPath;
}

float Costmap::aStarDist(vector<int> sLoc, vector<int> gLoc){
	float costMapPenalty = 0;
	vector<vector<int> > cSet; // 1 means in closed set, 0 means not
	vector<vector<int> > oSet; // 1 means in open set, 0 means not
	vector<vector<float> > gScore; // known cost from initial node to n
	vector<vector<float> > fScore; // gScore + heuristic score (dist to goal + imposed cost)
	for(int i=0;i<this->nRows; i++){
		vector<int> tC;
		vector<int> tO;
		vector<float> tG;
		vector<float> tF;
		for(int j=0; j<this->nCols; j++){
			tC.push_back(0);
			tO.push_back(0);
			tG.push_back(INFINITY);
			tF.push_back(INFINITY);
		}
		cSet.push_back(tC);
		oSet.push_back(tC);
		gScore.push_back(tG); // init scores to inf
		fScore.push_back(tF); // init scores to inf
	}
	oSet[sLoc[0]][sLoc[1]] = 1; // starting node has score 0
	gScore[sLoc[0]][sLoc[1]] = 0; // starting node in open set
	if(this->cells[sLoc[0]][sLoc[1]] < 3){
		costMapPenalty = this->cells[sLoc[0]][sLoc[1]];
	}
	else{
		costMapPenalty = INFINITY;
	}
	fScore[sLoc[0]][sLoc[1]] = gScore[sLoc[0]][sLoc[1]] + this->getEuclidDist(sLoc[0],sLoc[1],gLoc[0],gLoc[1]) + costMapPenalty;

	int foo = 1;
	int finishFlag = 0;
	while(foo>0 && finishFlag == 0){
		/////////////////// this finds node with lowest fScore and makes current
		float min = INFINITY;
		vector<int> iMin;
		iMin.push_back(0);
		iMin.push_back(0);
		for(int i=0; i<this->nRows; i++){
			for(int j=0; j<this->nCols; j++){
				if(oSet[i][j] > 0 && fScore[i][j] < min){
					min = fScore[i][j];
					iMin[0] = i;
					iMin[1] = j;
				}
			}
		}
		vector<int> cLoc = iMin;
		/////////////////////// end finding current node
		if(cLoc == gLoc){ // if the current node equals goal, construct path
			finishFlag = 1;
			return fScore[gLoc[0]][gLoc[1]];
		} ///////////////////////////////// end construct path
		oSet[cLoc[0]][cLoc[1]] = 0;
		cSet[cLoc[0]][cLoc[1]] = 1;
		for(int nbrRow=cLoc[0]-1;nbrRow<cLoc[0]+2;nbrRow++){
			for(int nbrCol=cLoc[1]-1; nbrCol<cLoc[1]+2; nbrCol++){
				if(cSet[nbrRow][nbrCol] == 1){ // has it already been eval? in cSet
					continue;
				}
				float tGScore;
				tGScore = gScore[cLoc[0]][cLoc[1]] + this->getEuclidDist(cLoc[0],cLoc[1],nbrRow,nbrCol); // calc temporary gscore
				if(oSet[nbrRow][nbrCol] == 0){
					oSet[nbrRow][nbrCol] = 1;  // add nbr to open set
				}
				else if(tGScore >= gScore[nbrRow][nbrCol]){ // is temp gscore better than stored g score of nbr
					continue;
				}
				gScore[nbrRow][nbrCol] = tGScore;
				if(this->cells[nbrRow][nbrCol] < 3){
					costMapPenalty = this->cells[nbrRow][nbrCol];
				}
				else{
					costMapPenalty = INFINITY;
				}
				fScore[nbrRow][nbrCol] = gScore[nbrRow][nbrCol] + this->getEuclidDist(gLoc[0],gLoc[1],nbrRow,nbrCol) + costMapPenalty;
			}
		}
		/////////////// end condition for while loop, check if oSet is empty
		foo = 0;
		for(int i=0; i<this->nRows; i++){
			for(int j=0; j<this->nCols; j++){
				foo+= oSet[i][j];
			}
		}
	}
	return 0;
}
