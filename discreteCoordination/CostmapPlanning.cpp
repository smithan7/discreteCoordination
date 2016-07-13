/*
 * CostmapPlanning.cpp
 *
 *  Created on: Jul 13, 2016
 *      Author: andy
 */

#include "CostmapPlanning.h"

CostmapPlanning::CostmapPlanning() {


}

CostmapPlanning::~CostmapPlanning() {}

void CostmapPlanning::findFrontiers(){
	vector<vector<int> > frontiersList;
	this->frntsExist = false;
	for(int i=1; i<costmap.nRows-1; i++){
		for(int j=1; j<costmap.nCols-1; j++){
			bool newFrnt = false;
			if(costmap.cells[i][j] > 100 && costmap.cells[i][j] < 110){ // i'm unobserved
				if(costmap.cells[i+1][j] < 10){ //  but one of my nbrs is observed
					newFrnt = true;
				}
				else if(costmap.cells[i-1][j] < 10){ //  but one of my nbrs is observed
					newFrnt = true;
				}
				else if(costmap.cells[i][j+1] < 10){ //  but one of my nbrs is observed
					newFrnt = true;
				}
				else if(costmap.cells[i][j-1] < 10){ //  but one of my nbrs is observed
					newFrnt = true;
				}
			}
			if(newFrnt){
				vector<int> fT;
				fT.push_back(i);
				fT.push_back(j);
				frontiersList.push_back(fT);
				this->frntsExist = true;
			}
		}
	}
	cout << "this->frontiers in: " << this->frontiers.size() << endl;
	for(size_t i=0; i<this->frontiers.size(); i++){
		cout << "   " << this->frontiers[i].centroid[0] << " / " << this->frontiers[i].centroid[1] << endl;
	}

	// check to see if frnt.centroid is still a Frontier cell, if so keep, else delete
	for(size_t i=0; i<this->frontiers.size(); i++){
		this->frontiers[i].editFlag = true;
		bool flag = true;
		for(int j=0; j<(int)frontiersList.size(); j++){
			if(this->frontiers[i].centroid == frontiersList[j]){
				flag = false;
				frontiersList.erase(frontiersList.begin()+j);
			}
		}
		if(flag){
			this->frontiers.erase(this->frontiers.begin()+i);
		}
		else{
			this->frontiers[i].editFlag = false;
		}
	}
	// breadth first search through known clusters
	for(size_t i=0; i<this->frontiers.size(); i++){ // keep checking for new Frontier clusters while there are unclaimed Frontiers
		vector<vector<int> > q; // current cluster
		vector<vector<int> > qP; // open set in cluster
		qP.push_back(this->frontiers[i].centroid);

		while((int)qP.size() > 0){ // find all nbrs of those in q
			vector<int> seed = qP[0];
			q.push_back(qP[0]);
			qP.erase(qP.begin(),qP.begin()+1);
			for(int ni = seed[0]-2; ni<seed[0]+3; ni++){
				for(int nj = seed[1]-2; nj<seed[1]+3; nj++){
					for(int i=0; i<(int)frontiersList.size(); i++){
						if(frontiersList[i][0] == ni && frontiersList[i][1] == nj){
							qP.push_back(frontiersList[i]); // in range, add to open set
							frontiersList.erase(frontiersList.begin() + i);
						}
					}
				}
			}
		}
		this->frontiers[i].members = q; // save to list of clusters
	}

	// breadth first search
	while((int)frontiersList.size() > 0){ // keep checking for new Frontier clusters while there are unclaimed Frontiers
		vector<vector<int> > q; // current cluster
		vector<vector<int> > qP; // open set in cluster
		qP.push_back(frontiersList[0]);
		frontiersList.erase(frontiersList.begin());

		while((int)qP.size() > 0){ // find all nbrs of those in q
			vector<int> seed = qP[0];
			q.push_back(qP[0]);
			qP.erase(qP.begin(),qP.begin()+1);
			for(int ni = seed[0]-1; ni<seed[0]+2; ni++){
				for(int nj = seed[1]-1; nj<seed[1]+2; nj++){
					for(int i=0; i<(int)frontiersList.size(); i++){
						if(frontiersList[i][0] == ni && frontiersList[i][1] == nj){
							qP.push_back(frontiersList[i]); // in range, add to open set
							frontiersList.erase(frontiersList.begin() + i, frontiersList.begin()+i+1);
						}
					}
				}
			}
		}
		Frontier a(q);
		this->frontiers.push_back(a);
	}

	for(size_t i=0; i<this->frontiers.size(); i++){ // number of clusters
		if(this->frontiers[i].editFlag){
			float minDist = INFINITY;
			int minDex;
			for(size_t j=0; j<this->frontiers[i].members.size(); j++){ // go through each cluster member
				int jx = this->frontiers[i].members[j][0];
				int jy = this->frontiers[i].members[j][1];
				float tempDist = 0;
				for(size_t k=0; k<this->frontiers[i].members.size(); k++){ // and get cumulative distance to all other members
					int kx = this->frontiers[i].members[k][0];
					int ky = this->frontiers[i].members[k][1];
					tempDist += this->getEuclidDist(jx, jy, kx, ky);
				}
				if(tempDist < minDist){
					minDist = tempDist;
					minDex = j;
				}
			}
			this->frontiers[i].centroid = this->frontiers[i].members[minDex];
		}
	}
}

