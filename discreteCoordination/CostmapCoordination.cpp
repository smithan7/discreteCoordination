/*
 * CostmapCoordination.cpp
 *
 *  Created on: Jul 12, 2016
 *      Author: andy
 */

#include "CostmapCoordination.h"

CostmapCoordination::CostmapCoordination(vector<vector<float> > euclidDist){
	this->euclidDist = euclidDist;
}

CostmapCoordination::~CostmapCoordination() {}

vector<int> CostmapCoordination::getGoal(vector<int> cLoc, Costmap &costmap){

	clusterFrontiers(this->findFrontiers(costmap)); // finds and clusters frontiers into frontiers class
	getFrontierCosts(cLoc, costmap); // find distance to frontiers

	return marketFrontiers();
}

void CostmapCoordination::getFrontierCosts(vector<int> cLoc, Costmap &costmap){
	for(size_t i=0; i<this->frontiers.size(); i++){
		this->frontiers[i].cost = costmap.aStarDist(cLoc, this->frontiers[i].centroid);
	}
}

vector<vector<int> > CostmapCoordination::findFrontiers(Costmap &costmap){
	vector<vector<int> > frontiersList;
	for(int i=1; i<costmap.nCols-1; i++){
		for(int j=1; j<costmap.nRows-1; j++){
			bool newFrnt = false;
			if(costmap.cells[i][j] == 101){ // i'm unobserved
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
			}
		}
	}
	return frontiersList;
}

void CostmapCoordination::clusterFrontiers(vector<vector<int> >  frntList){
	// check to see if frnt.centroid is still a Frontier cell, if so keep, else delete
	for(size_t i=0; i<this->frontiers.size(); i++){
		this->frontiers[i].editFlag = true;
		bool flag = true;
		for(size_t j=0; j<frntList.size(); j++){
			if(this->frontiers[i].centroid == frntList[j]){
				flag = false;
				frntList.erase(frntList.begin()+j);
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
					for(size_t i=0; i<frntList.size(); i++){
						if(frntList[i][0] == ni && frntList[i][1] == nj){
							qP.push_back(frntList[i]); // in range, add to open set
							frntList.erase(frntList.begin() + i);
						}
					}
				}
			}
		}
		this->frontiers[i].members = q; // save to list of clusters
	}

	// breadth first search
	while(frntList.size() > 0){ // keep checking for new Frontier clusters while there are unclaimed Frontiers
		vector<vector<int> > q; // current cluster
		vector<vector<int> > qP; // open set in cluster
		qP.push_back(frntList[0]);
		frntList.erase(frntList.begin());

		while((int)qP.size() > 0){ // find all nbrs of those in q
			vector<int> seed = qP[0];
			q.push_back(qP[0]);
			qP.erase(qP.begin(),qP.begin()+1);
			for(int ni = seed[0]-1; ni<seed[0]+2; ni++){
				for(int nj = seed[1]-1; nj<seed[1]+2; nj++){
					for(int i=0; i<(int)frntList.size(); i++){
						if(frntList[i][0] == ni && frntList[i][1] == nj){
							qP.push_back(frntList[i]); // in range, add to open set
							frntList.erase(frntList.begin() + i, frntList.begin()+i+1);
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


vector< vector<int> > CostmapCoordination::kMeansClusteringEuclid(int numClusters, Costmap &costmap){
	vector<Frontier> tempFrnt = this->frontiers;
	vector<int> cluster[numClusters]; // array of vectors
	float bestClusterDist = INFINITY;
	vector< vector<int> > bestClusterSet;
	for(int i=0; i<numClusters; i++){
		vector<int> temp;
		bestClusterSet.push_back(temp);
	}
	bool convergeFlag = false;
	bool initFlag = true;
	while(initFlag){
		for(int i=0; i<numClusters; i++){ // generate one random cluster per UAV
			if(tempFrnt.size() == 0){
				initFlag = false;
			}
			else{
				int temp = rand() % tempFrnt.size();
				// TODO this was the node I think and I changed to index of 'frontiers'
				cluster[i].push_back(temp); // assign random cluster centroid initially
				//
				tempFrnt.erase(tempFrnt.begin()+temp); // don't allow center to be taken
			}
		}
	}

	tempFrnt = this->frontiers;
	while(convergeFlag == false){ // until the solution converges
		float tempDist = 0;
		for(int i=0; i<numClusters; i++){ // compute centroid Frontier of each cluster
			if(cluster[i].size() > 2){ // no point if a cluster has 1 or 2 members
				float minDist[2] = {INFINITY, -1};
				for(int j=0; j<int(cluster[i].size()); j++){ // find A* dist from each node in each cluster
					float distSum = 0;
					for(int k=0; k<int(cluster[i].size()); k++){ // to each other node in that cluster
						distSum += costmap.euclidDist[ cluster[i][j] ][ cluster[i][k] ];
					} // end to each other node
					if(distSum < minDist[0]){ // is it the most central
						minDist[0] = distSum;
						minDist[1] = cluster[i][j]; // assign as new centroid
					} // end is it most central
				} // end find A* dist from each node in each cluster
				cluster[i].erase(cluster[i].begin(), cluster[i].end()); // erase cluster
				cluster[i].push_back(minDist[1]); // set the new cluster centroid
				tempDist += minDist[0];
			} // end if has more than 2 members
			else if(cluster[i].size() == 2){ // if it has two members calc distance between them
				cluster[i].erase(cluster[i].begin()+1); // erase cluster
				tempDist += costmap.getEuclidDist(cluster[i][0][0],  ][ cluster[i][1] ];
			} // end if it has two members calc distance
			else{ // this cluster has one member
				tempDist += 0;
			} // end this cluster has one member
		} // end compute centroid Frontier of each cluster

		for(int i=0; i<int(openFrnt.size()); i++){ // find A* dist from each Frontier to each cluster and group Frontier in closest cluster
			float minDist[2] = {INFINITY, -1}; // [dist, index];
			bool isCenter = false;
			for(int j=0; j<numClusters; j++){ // find closest cluster center to current front
				if(openFrnt[i] == cluster[j][0]){
					isCenter = true;
				}
			}
			if(!isCenter){ // is it a centroid
				for(int j=0; j<numClusters; j++){ // find closest cluster center to current front
					float tempDist = this->distGraph[ openFrnt[i]] [cluster[j][0]]; // calc dist between
					if(tempDist < minDist[0]){ // new centroid is closer, so switch
						minDist[0] = tempDist;
						minDist[1] = j; // closest centroid
					} // end new centroid is closer
				} // end find closest cluster center to current front
				if(minDist[1] >= 0){
					cluster[int(minDist[1])].push_back(openFrnt[i]); // add to appropriate cluster
				}
			} // end is it a centroid
		} // end check each cluster centroid
		if(tempDist < bestClusterDist){ // found a better solution, has not converged yet
			bestClusterDist = tempDist;
			for(int i=0; i<numClusters; i++){
				bestClusterSet[i].erase(bestClusterSet[i].begin(),bestClusterSet[i].end());
				for(int j=0; j<int(cluster[i].size()); j++){
					bestClusterSet[i].push_back(cluster[i][j]);
				}
			}
			convergeFlag = false;
		}
		else{ // it has converged
			convergeFlag = true;
		}
	} // end until solution converges
	return(bestClusterSet);
}

vector< vector<int> > CostmapCoordination::kMeansClusteringTravel(int numClusters, Costmap &costmap){
	vector<int> tempFrnt = openFrnt;
	vector<int> cluster[numClusters]; // array of vectors
	float bestClusterDist = INFINITY;
	vector< vector<int> > bestClusterSet;
	for(int i=0; i<numClusters; i++){
		vector<int> temp;
		bestClusterSet.push_back(temp);
	}
	bool convergeFlag = false;
	bool initFlag = true;
	while(initFlag){
		for(int i=0; i<numClusters; i++){ // generate one random cluster per UAV
			if(tempFrnt.size() == 0){
				initFlag = false;
			}
			else{
				int temp = rand() % tempFrnt.size();
				cluster[i].push_back(tempFrnt[temp]); // assign random cluster centroid initially
				tempFrnt.erase(tempFrnt.begin()+temp); // don't allow center to be taken
			}
		}
	}

	tempFrnt = openFrnt;
	while(convergeFlag == false){ // until the solution converges
		float tempDist = 0;
		for(int i=0; i<numClusters; i++){ // compute centroid Frontier of each cluster
			if(cluster[i].size() > 2){ // no point if a cluster has 1 or 2 members
				float minDist[2] = {INFINITY, -1};
				for(int j=0; j<int(cluster[i].size()); j++){ // find A* dist from each node in each cluster
					float distSum = 0;
					for(int k=0; k<int(cluster[i].size()); k++){ // to each other node in that cluster
						distSum += this->aStarDist(cluster[i][j],cluster[i][k]);
					} // end to each other node
					if(distSum < minDist[0]){ // is it the most central
						minDist[0] = distSum;
						minDist[1] = cluster[i][j]; // assign as new centroid
					} // end is it most central
				} // end find A* dist from each node in each cluster
				cluster[i].erase(cluster[i].begin(), cluster[i].end()); // erase cluster
				cluster[i].push_back(minDist[1]); // set the new cluster centroid
				tempDist += minDist[0];
			} // end if has more than 2 members
			else if(cluster[i].size() == 2){ // if it has two members calc distance between them
				cluster[i].erase(cluster[i].begin()+1); // erase cluster
				tempDist += this->aStarDist(cluster[i][0],cluster[i][1]);
			} // end if it has two members calc distance
			else{ // this cluster has one member
				tempDist += 0;
			} // end this cluster has one member
		} // end compute centroid Frontier of each cluster

		for(int i=0; i<int(openFrnt.size()); i++){ // find A* dist from each Frontier to each cluster and group Frontier in closest cluster
			float minDist[2] = {INFINITY, -1}; // [dist, index];
			bool isCenter = false;
			for(int j=0; j<numClusters; j++){ // find closest cluster center to current front
				if(openFrnt[i] == cluster[j][0]){
					isCenter = true;
				}
			}
			if(!isCenter){ // is it a centroid
				for(int j=0; j<numClusters; j++){ // find closest cluster center to current front
					float tempDist = this->aStarDist(openFrnt[i], cluster[j][0]); // calc dist between
					if(tempDist < minDist[0]){ // new centroid is closer, so switch
						minDist[0] = tempDist;
						minDist[1] = j; // closest centroid
					} // end new centroid is closer
				} // end find closest cluster center to current front
				if(minDist[1] >= 0){
					cluster[int(minDist[1])].push_back(openFrnt[i]); // add to appropriate cluster
				}
			} // end is it a centroid
		} // end check each cluster centroid
		if(tempDist < bestClusterDist){ // found a better solution, has not converged yet
			bestClusterDist = tempDist;
			for(int i=0; i<numClusters; i++){
				bestClusterSet[i].erase(bestClusterSet[i].begin(),bestClusterSet[i].end());
				for(int j=0; j<int(cluster[i].size()); j++){
					bestClusterSet[i].push_back(cluster[i][j]);
				}
			}
			convergeFlag = false;
		}
		else{ // it has converged
			convergeFlag = true;
		}
	} // end until solution converges
	return(bestClusterSet);
}

vector<vector<int> > CostmapCoordination::centralMarket(Costmap &costmap, vector<vector<int> > cLoc){

	cerr << "into get Frontier costs: " << endl;
	this->getFrontierCosts(gMap, bots);
	cerr << "out of frotier costs" << endl;

	cout << "Frontier costs and Rewards: " << endl;
	for(size_t i=0; i<this->frontiers.size(); i++){
		for(size_t j=0; j<bots.size(); j++){
			cout << "   cost: " << bots[j].fCost[i] << endl;;
			cout << "   reward: " << this->frontiers[i].reward << endl;
		}
	}
	waitKey(1);

	if(bots.size() <= this->frontiers.size()){ // more Frontiers than Agents
		cout << "more Frontiers than Agents" << endl;
		vector<vector<float> > fValueList; // list of Frontier values for all Agents, [Agent][Frontier]
		for(int i=0; i<bots.size(); i++){
			vector<float> cVal;
			for(int j=0; j<this->frontiers.size(); j++){
				cVal.push_back( this->frontiers[i].reward - bots[i].fCost[j] );
			}
			fValueList.push_back( cVal );
		}

		cout << "fValueList: " << endl;
		for(int i=0; i<bots.size(); i++){
			cout << "   ";
			for(int j=0; j<this->frontiers.size(); j++){
				cout << fValueList[i][j] << " , ";
			}
			cout << endl;
		}

		bool fin = false;
		vector<int> maxDex;
		vector<float> maxVal;

		while(!fin){
			maxDex.erase(maxDex.begin(), maxDex.end());
			maxVal.erase(maxVal.begin(), maxVal.end());
			fin = true;

			for(int i=0; i<bots.size(); i++){ // get each Agents best Frontier
				maxDex.push_back( -1 );
				maxVal.push_back( -INFINITY );

				for(int j=0; j<(int)fValueList[i].size(); j++){
					if(fValueList[i][j] > maxVal[i]){
						maxDex[i] = j; // Agent's max value is Frontier j
						maxVal[i] = fValueList[i][j];
					}
				}
			}

			// make sure no one shares the same Frontier
			for(int i=0; i<bots.size(); i++){
				for(int j=i+1; j<bots.size(); j++){
					if(i!=j && maxDex[i] == maxDex[j]){ // not me and has the same goal;
						fin = false;
						if(maxVal[i] >= maxVal[j]){
							fValueList[j][maxDex[j]] = -INFINITY;
						}
						else{
							fValueList[i][maxDex[i]] = -INFINITY;
						}
					}
				}
			}
		}
		for(int i=0; i<bots.size(); i++){
			bots[i].gLoc = this->frontiers[maxDex[i]].centroid;
			bots[i].gIndex = maxDex[i];
		}
	}
	else{ // more Agents than Frontiers
		cout << "more Agents than Frontiers" << endl;
		for(int i=0; i<this->frontiers.size(); i++){ // go through all Frontiers and find the best Agent
			float mV = INFINITY;
			int mI;
			for(int j=0; j<bots.size(); j++){
				if(bots[j].fCost[i] < mV){
					mV = bots[j].fCost[i];
					mI = j;
				}
			}
			bots[mI].gLoc = this->frontiers[i].centroid;
			bots[mI].gIndex = i;
			for(int j = 0; j<(int)this->frontiers.size(); j++){ // erase all of the value for the worst Agent
				bots[mI].fCost[j] = INFINITY;
			}
		}
	}
}
