/*
 * MiniGraph.cpp
 *
 *  Created on: Mar 14, 2016
 *      Author: andy
 */

#include "MiniGraph.h"

using namespace cv;
using namespace std;

MiniGraph::MiniGraph(){
	// TODO Auto-generated constructor stub
	int obsRadius = 40;
	this->grafSpacing = 3;
	this->grafConnectionSpacing = 6;
	Mat temp =Mat::zeros(2*(obsRadius + 1), 2*(obsRadius + 1), CV_8UC1);
	Scalar white = 255;
	circle(temp,Point{obsRadius, obsRadius},obsRadius, white);

	for(int i=0; i<temp.cols; i++){
		for(int j=0; j<temp.rows; j++){
			if(temp.at<uchar>(i,j,0) == 255){
				vector<int> t;
				t.push_back(i-obsRadius);
				t.push_back(j-obsRadius);
				this->viewPerim.push_back(t);
			}
		}
	}
}

int MiniGraph::masterGraphPathPlanning(float maxLength, int nPullsTSP, int nPullsEvolveMasterNodes){

	vector<Mat> observations;
	for(size_t i=0; i<graf.size(); i++){
		Mat t = Mat::zeros(this->freeMat.size(), CV_8UC1);
		observations.push_back(t);
		simulateObservation(i, observations[i]);
	}
	vector<int> masterNodes; // master node indices to get back to graf
	vector<float> masterRewards; // reward of each master node
	cerr << "into masternodes evolution" << endl;
	this->findMasterNodesEvolution(observations, masterNodes, masterRewards, nPullsEvolveMasterNodes); // find master nodes

	Mat view = Mat::zeros(observations[0].size(), CV_8UC1);
	for(size_t i=0; i<masterNodes.size(); i++){
		bitwise_or(view, observations[masterNodes[i]], view);
	}
	for(size_t i=0; i<masterNodes.size(); i++){
		Point t;
		t.x = this->graf[masterNodes[i]][1];
		t.y = this->graf[masterNodes[i]][0];
		circle(view, t, 2, Scalar(127), -1);
	}

	Point tp;
	tp.x = this->graf[this->cNode][1];
	tp.y = this->graf[this->cNode][0];
	circle(view, tp, 5, Scalar(200), 2);


	cerr << "masterNodes.size(): " << masterNodes.size() << endl;
	namedWindow("nodes view", WINDOW_NORMAL);
	imshow("nodes view", view);
	waitKey(1);

	cerr << "out of masterNodes evolution" << endl;
	vector<vector<float> > masterGraph; // A* dist from / to all master nodes
	this->buildMasterGraph(masterNodes, masterGraph); // get A* dist
	cerr << "out of buildMasterGraph" << endl;
	vector<int> tspPath = this->tspMasterGraphPathPlanner(masterNodes, masterGraph, masterRewards, nPullsTSP, maxLength);
	cerr << "out of tspMasterGraphPathPlanner" << endl;

	Mat tMat = Mat::zeros(observations[0].size(), CV_8UC1);;
	for(size_t i=0; i<masterNodes.size(); i++){
		cerr << "masterNode / observations: " << masterNodes[i] << " / " << observations.size() << endl;
		bitwise_or(tMat, observations[masterNodes[i]], tMat);
	}

	cerr << "tspPath[" << tspPath.size() << "]: " << tspPath[0];
	for(size_t i=1; i<tspPath.size(); i++){
		cerr << ", " << tspPath[i];
		Point t;
		t.x = this->graf[tspPath[i]][1];
		t.y = this->graf[tspPath[i]][0];
		Point t2;
		t2.x = this->graf[tspPath[i-1]][1];
		t2.y = this->graf[tspPath[i-1]][0];
		line(tMat, t, t2, Scalar(127), 1);
	}
	cerr << endl;

	for(size_t i=0; i<masterNodes.size(); i++){
		Point t;
		t.x = this->graf[masterNodes[i]][1];
		t.y = this->graf[masterNodes[i]][0];
		circle(tMat, t, 2, Scalar(127), -1);
	}

	Point t;
	t.x = this->graf[tspPath[1]][1];
	t.y = this->graf[tspPath[1]][0];
	circle(tMat, t, 2, Scalar(200), 3);

	namedWindow("ws view", WINDOW_NORMAL);
	imshow("ws view", tMat);
	waitKey(1);

	cerr << "returning tspPath[1]" << endl;
	return tspPath[1];
}


void MiniGraph::mctsPathPlanner(vector<int> &path, float maxLength, int maxPulls){
	 vector<float> rewards;
	vector<Mat> observations;
	for(size_t i=0; i<graf.size(); i++){
		Mat t = Mat::zeros(this->freeMat.size(), CV_8UC1);
		observations.push_back(t);
		simulateObservation(i, observations[i]);
	}
	//cerr << "mctsPathPlanner::making root node" << endl;
	TreeNode myTree(this->cNode, this->distGraph, observations, path, 0, 0);
	while(myTree.nPulls < maxPulls){
		//cerr << "mctsPathPlanner::into searchTree" << endl;
		myTree.searchTree(this->distGraph, observations, maxLength);
		//cerr << "out of searchTree" << endl;
	}
	myTree.exploitTree(path, rewards);
	//cerr << "out of loop" << endl;

	cerr << "mctsPath: ";
	for(size_t i=0; i<path.size(); i++){
		cerr << path[i] << ", ";
	}
	cerr << endl;
	cerr << "mctsReward: ";
	for(size_t i=0; i<rewards.size(); i++){
		cerr << rewards[i] << ", ";
	}
	cerr << endl;
	waitKey(1);
}

float MiniGraph::matReward(Mat &in){
	// get Mat entropy
	float observed = 0;
	for(int i=0; i<in.cols; i++){
		for(int j=0; j<in.rows; j++){
			if(in.at<uchar>(i,j,0) > 0){
				observed++;
			}
		}
	}
	return observed;
}

void MiniGraph::findMasterNodesEvolution(vector<Mat> observations, vector<int> &masterNodes, vector<float> &masterRewards, int nPulls){

	vector<int> nodeList;
	vector<float> nodeRewards;

	vector<int> openNodes;
	Mat gView = Mat::zeros(observations[0].size(), CV_8UC1);
	for(size_t i=0; i<observations.size(); i++){
		nodeRewards.push_back(this->matReward(observations[i]));
		openNodes.push_back(i);
		nodeList.push_back(i);
		bitwise_or(gView, observations[i], gView);
	}

	float gReward = this->matReward(gView);

	vector<int> bestNodes;
	vector<int> workingNodes;

	vector<int> cNodes;
	float cReward = 0;
	cerr << "2" << endl;
	Mat cView = Mat::zeros(observations[0].size(), CV_8UC1);
	while(cReward < gReward * 0.95){
		int c = rand() % openNodes.size();
		cNodes.push_back(openNodes[c]);
		bitwise_or(cView, observations[openNodes[c]], cView);
		cReward = this->matReward(cView);
		openNodes.erase(openNodes.begin() + c);
	}
	cerr << "3" << endl;

	workingNodes = cNodes;
	bestNodes = cNodes;

	vector<int> tempList;
	for(int i=0; i<nPulls; i++){
		tempList = openNodes;
		cerr << "a" << tempList.size() << " < " << cNodes.size() << endl;
		if(cNodes.size() == 0){
			waitKey(0);
		}
		int oni = rand() % tempList.size();
		int cni = rand() % cNodes.size();
		cerr << "a.5" << endl;
		if(  rand() % 1000 > 500 ){ // swap a node with open set
			cerr << "b" << endl;
			int swpIn = tempList[oni];
			int swpOut = cNodes[cni];

			cNodes.erase(cNodes.begin() + cni);
			cNodes.push_back(swpIn);
			tempList.erase(tempList.begin() + oni);
			tempList.push_back(swpOut);
		}
		else{ // erase a node
			cerr << "c" << endl;
			int swap = cNodes[cni];
			cNodes.erase(cNodes.begin()+cni);
			tempList.push_back(swap);
		}
		cerr << "d" << endl;
		Mat cView = Mat::zeros(observations[0].size(), CV_8UC1);
		for(size_t i=0; i<cNodes.size(); i++){
			bitwise_or(cView, observations[cNodes[i]], cView);
		}
		cerr << "e" << endl;
		cReward = this->matReward(cView);
		cerr << "f" << endl;
		bool flag = false;

		if(cReward > 0.95 * gReward){
			cerr << "g" << endl;
			if(cNodes.size() < bestNodes.size()){
				bestNodes = cNodes;
			}
			if( cNodes.size() < workingNodes.size()){
				workingNodes = cNodes;
				flag = true;
			}
			else if( rand() % 1000 > 500 ){
				workingNodes = cNodes;
				flag = true;
			}
		}
		cerr << "h" << endl;
		if(flag){
			openNodes = tempList;
		}
		cNodes = workingNodes;
	}

	cerr << "4" << endl;

	Mat bView = Mat::zeros(observations[0].size(), CV_8UC1);
	for(size_t i=0; i<bestNodes.size(); i++){
		bitwise_or(bView, observations[bestNodes[i]], bView);
	}
	cerr << "bestNodes.size(): " << bestNodes.size() << endl;
	cReward = this->matReward(bView);

	cerr << "gReward: " << gReward << endl;
	cerr << "cReward: " << cReward << endl;

	if(bestNodes.size() < 2){
		bestNodes.push_back(openNodes[0]);
	}

	masterNodes.clear(); // ensure I start with a fresh list
	masterRewards.clear(); // ensure I start with a fresh list
	masterNodes.push_back(this->cNode);
	masterRewards.push_back(0);
	for(size_t i=0; i<bestNodes.size(); i++){
		masterNodes.push_back(bestNodes[i]);
		masterRewards.push_back(nodeRewards[bestNodes[i]]);
	}

	cerr << "5" << endl;

}

void MiniGraph::buildMasterGraph(vector<int> &masterNodes, vector<vector<float> > &masterGraph){
	for(size_t i =0; i<masterNodes.size(); i++){
		vector<float> tg;
		for(size_t j =0; j<masterNodes.size(); j++){
			if(i != j){
				tg.push_back(this->aStarDist(masterNodes[i], masterNodes[j]));
			}
			else{
				tg.push_back(INFINITY);
			}
		}
		masterGraph.push_back(tg);
	}
}


vector<int> MiniGraph::tspMasterGraphPathPlanner(vector<int> &masterNodes, vector<vector<float> > &masterTrans, vector<float> &rewards, int maxPulls, float maxLength){
	vector<int> bestPath = tspMasterGraphBuildPath(masterTrans);
	float bestReward = 0;
	vector<int> workingPath = tspMasterGraphBuildPath(masterTrans);
	float workingReward = 0;
	float temp = 1000;
	for(int i=0; i<maxPulls; i++){
		vector<int> tempPath = this->tspMasterGraphEvolvePath(workingPath);
		float tempPathReward = this->tspMasterGraphEvaluatePath(tempPath, rewards, masterTrans, maxLength);

		float p = exp( (workingReward - tempPathReward)/temp );
		float pathLength = 0;
		for(size_t i=1; i<tempPath.size(); i++){
			pathLength += masterTrans[tempPath[i]][tempPath[i-1]];
		}

		if(tempPathReward > workingReward){
			workingReward = tempPathReward;
			workingPath = tempPath;
		}
		else if( exp( (workingReward - tempPathReward)/temp ) > (rand() % 1000)/1000 ){
			workingReward = tempPathReward;
			workingPath = tempPath;
		}
		if(tempPathReward > bestReward){
			bestReward = tempPathReward;
			bestPath = tempPath;
		}
		temp = temp * 0.9995;
	}
	// at end convert back to masternodes to get space to put back in graf
	vector<int> outPath;
	for(size_t i=0; i<bestPath.size(); i++){
		outPath.push_back(masterNodes[bestPath[i]]);
	}
	return outPath;
}

vector<int> MiniGraph::tspMasterGraphEvolvePath(vector<int> inPath){
	vector<int> path = inPath;
	if(path.size() > 2){
		int p = rand() % 10000;
		if(p > 5000){ // swap two
			for(int i=0; i<2; i++){
				int c = 1+ rand() % (path.size()-1);
				int c2 = c;
				while(c == c2){
					c2 = 1 + rand() % (path.size()-1);
				}
				int c3 = path[c];
				path[c] = path[c2];
				path[c2] = c3;
			}
		}
		else{ // swap 1
			int c = 1+ rand() % (path.size()-1);
			int c2 = c;
			while(c == c2){
				c2 = 1 + rand() % (path.size()-1);
			}
			int c3 = path[c];
			path[c] = path[c2];
			path[c2] = c3;
		}
	}
	return path;
}


vector<int> MiniGraph::tspMasterGraphBuildPath(vector<vector<float> > &masterTrans){
	vector<int> path;
	for(size_t i=0; i<masterTrans.size(); i++){
		path.push_back(i);
	}
	return path;
}

float MiniGraph::tspMasterGraphEvaluatePath(vector<int> &path, vector<float> &rewards, vector<vector<float> > &masterTrans, float maxLength){
	float pathReward = 0;
	float pathLength = 0;
	int i=1;
	bool flag = true;
	while(flag && i < masterTrans.size()){
		pathLength += masterTrans[path[i]][path[i-1]];
		if(pathLength < maxLength){
			pathReward += rewards[path[i]];
		}
		else{
			pathLength -= masterTrans[path[i]][path[i-1]];
			flag = false;
		}
		i++;
	}
	return pathReward - pathLength;
}

void MiniGraph::findMasterNodesDomination(vector<Mat> observations, vector<int> &masterNodes, vector<float> &masterRewards){

	masterNodes.clear(); // ensure I start with a fresh list

	// find nodes that are dominated by a nbr, their nbr covers > 90% their view + some extra
	vector<float> nodeRewards;
	for(size_t i=0; i<observations.size(); i++){
		nodeRewards.push_back(this->matReward(observations[i]));
	}

	for(size_t i=0; i<this->distGraph.size(); i++){ // check all nodes
		bool notDominated = true;
		Mat invMyObs;
		bitwise_not(observations[i], invMyObs);

		if(nodeRewards[i] > 0){ // does it have a useable view
			for(size_t j=0; j<this->distGraph[i].size(); j++){ // check against all other nodes
				if(this->distGraph[i][j] < this->grafConnectionSpacing *1000 && nodeRewards[j] > nodeRewards[i]){ // is it a nbr and does it have a view better than mine

					Mat invNbrObs;
					bitwise_not(observations[j], invNbrObs);

					// get cells I observe they dont
					Mat myContMat;
					bitwise_and(invNbrObs, observations[i], myContMat);
					float myContribution = this->matReward(myContMat);
					// get cells they observe I dont
					Mat nbrContMat;
					bitwise_and(invMyObs, observations[j], nbrContMat);
					float nbrContribution = this->matReward(nbrContMat);

					if(nbrContribution > myContribution){
						notDominated = false;
					}
				}
			}
		}
		else{ // I have no reward, dominated
			notDominated = false;
		}
		if(notDominated){
			masterNodes.push_back(i);
			masterRewards.push_back(nodeRewards[i]);
		}
		else{ // means I am dominated
			observations[i] = Mat::zeros(observations[i].size(), CV_8UC1); // clear view
		}
	}
}

vector<int> MiniGraph::tspPathPlanner(float maxDist, int maxPulls){
	//cerr << "tsp:1" << endl;
	vector<Mat> observations;
	for(size_t i=0; i<graf.size(); i++){
		Mat t = Mat::zeros(this->freeMat.size(), CV_8UC1);
		observations.push_back(t);
		simulateObservation(i, observations[i]);
	}

	vector<int> masterNodes;
	vector<float> masterRewards;
	this->findMasterNodesEvolution(observations, masterNodes, masterRewards, 10000);
	//this->findMasterNodesDomination(observations, masterNodes, masterRewards);


	Mat tMat = Mat::zeros(observations[0].size(), CV_8UC1);;
	for(size_t i=0; i<masterNodes.size(); i++){
		bitwise_or(tMat, observations[masterNodes[i]], tMat);
	}

	for(size_t i=0; i<masterNodes.size(); i++){
		Point t;
		t.x = this->graf[masterNodes[i]][1];
		t.y = this->graf[masterNodes[i]][0];
		circle(tMat, t, 2, Scalar(127), -1);
	}
	cerr << "graf / ws sizes: " << observations.size() << " / " << masterNodes.size() << endl;

	namedWindow("ws view", WINDOW_NORMAL);
	imshow("ws view", tMat);

	vector<vector<float> > masterGraph;
	this->buildMasterGraph(masterNodes, masterGraph);

	for(size_t i=0; i<masterGraph.size(); i++){
		for(size_t j=0; j<masterGraph[i].size(); j++){
			cerr << masterGraph[i][j] << ", ";
		}
		cerr << endl;
	}

	waitKey(1);

	//cerr << "tsp:2" << endl;
	vector<int> bestPath = this->buildPath(maxDist,masterNodes,masterGraph);
	//cerr << "tsp:2.5" << endl;
	float bestPathReward = this->getPathReward(bestPath, observations);
	//cerr << "tsp:3" << endl;
	vector<int> curPath = bestPath;
	float curPathReward = bestPathReward;

	float temp = 1000;
	for(int i=0; i<maxPulls; i++){
		//cerr << "tsp:4" << endl;
		vector<int> tempPath = this->modifyPath(bestPath, maxDist, masterNodes, masterGraph);
		//cerr << "tsp:4.5" << endl;
		float tempPathReward = this->getPathReward(tempPath, observations);
		//cerr << "tsp:5: " << tempPathReward << endl;
		if(tempPathReward > curPathReward){
			curPathReward = tempPathReward;
			curPath = tempPath;
		}
		else if( exp( (curPathReward - tempPathReward)/temp) > (rand() % 1000)/1000 ){
			curPathReward = tempPathReward;
			curPath = tempPath;
		}

		if(tempPathReward > bestPathReward){
			bestPathReward = tempPathReward;
			bestPath = tempPath;
		}
		temp = temp * 0.995;
		//cerr << "tsp:6: " << i << endl;
	}
	return bestPath;
}

vector<int> MiniGraph::buildPath(float maxDist, vector<int> &masterNodes, vector<vector<float> > &masterGraph){
	vector<int> path;
	path.push_back(this->cNode);
	float remDist = maxDist;
	int cur = this->cNode;
	while(remDist > 0){
		int nbr = this->getRandomNbr(cur, remDist, masterGraph);
		if(nbr >= 0){
			remDist = remDist - this->distGraph[cur][nbr];
			path.push_back(nbr);
			cur = nbr;
		}
		else{
			remDist = -1;
		}
	}
	return path;
}

vector<int> MiniGraph::modifyPath(vector<int> bestPath, float maxDist, vector<int> &masterNodes, vector<vector<float> > &masterGraph){
	cerr << "bestPath: ";
		for(size_t i=0; i<bestPath.size(); i++){
			cerr << bestPath[i] << ", ";
		}
		cerr << endl;


	int cur = 1;
	cerr << "modifyPath:a" << endl;
	if(bestPath.size() > 1){
		cur = 1 + (rand() % (bestPath.size()-1));
	}

	cerr << "modifyPath:b: " << cur << endl;
	vector<int> path;
	float remDist = maxDist;
	cerr << "modifyPath:c" << endl;
	for(size_t i=0; i<cur ; i++){
		path.push_back(bestPath[i]);
		remDist -= masterGraph[bestPath[cur]][bestPath[cur-1]];
	}


	cerr << "    Path: ";
	for(size_t i=0; i<path.size(); i++){
		cerr << path[i] << ", ";
	}
	cerr << endl;

	cur = path[path.size()-1];

	cerr << "modifyPath:d" << endl;
	while(remDist > 0){
		cerr << "modifyPath:e" << endl;
		int nbr = this->getRandomNbr(cur, remDist, masterGraph);
		cerr << "modifyPath:f" << endl;
		if(nbr >= 0){
			cerr << "modifyPath:g" << endl;
			remDist = remDist - masterGraph[cur][nbr];
			path.push_back(nbr);
			cur = nbr;
		}
		else{
			remDist = -1;
		}
		cerr << "modifyPath:h" << endl;
	}
	cerr << "modifyPath:i" << endl;
	return path;
}

float MiniGraph::getPathReward(vector<int> path, vector<Mat> &observations){
	//cerr << "getPathReward:a" << endl;
	Mat pathMat = Mat::zeros(this->freeMat.size(), CV_8UC1);
	//cerr << "getPathReward:b: " << path.size() << endl;
	for(size_t i=0; i<path.size()-1; i++){
		//cerr << "getPathReward:c: " << path[i] << endl;
		bitwise_or(pathMat,observations[path[i]], pathMat);
		//cerr << "getPathReward:d" << endl;
	}
	//cerr << "getPathReward:e" << endl;
	float reward = 0;
	for(size_t i=0; i<pathMat.cols; i++){
		for(size_t j=0; j<pathMat.rows; j++){
			if(pathMat.at<uchar>(i,j,0) == 255){
				reward++;
			}
		}
	}
	//cerr << "getPathReward:f" << endl;
	return reward;
}


int MiniGraph::getRandomNbr(int node, float remDist, vector<vector<float> > &masterGraph){
	vector<int> nbrs;
	for(size_t i=0; i<masterGraph[node].size(); i++){
		float d = masterGraph[node][i];
		if(d < remDist && d > 0){
			nbrs.push_back(i);
		}
	}
	if(nbrs.size() > 0){
		int c = nbrs[rand() % nbrs.size()];
		return c;
	}
	else{
		return -1;
	}
}

bool MiniGraph::lineTraversabilityCheck(Mat &tSpace, vector<int> sPt, vector<int> fPt, int fValue){

	if(abs(fPt[0] - sPt[0]) == abs(fPt[1] - sPt[1])){ // larger change in x direction, count along x
		if(fPt[0] < sPt[0]){ // set order right
			vector<int> t = sPt;
			sPt = fPt;
			fPt = t;
			cout << "inv" << endl;
		}

		float m = float(fPt[1] - sPt[1]) / float(fPt[0] - sPt[0]);
		float b = float(fPt[1]) - m*float(fPt[0]);

		cout << "fPt: " << fPt[1] << " , " << fPt[0] << endl;
		cout << "sPt: " << sPt[1] << " , " << sPt[0] << endl;
		cout << "x: " << m << " , " << b << endl;

		Mat temp = tSpace;

		for(int x = sPt[0]+1; x<fPt[0]-1; x++){
			float tx = x;
			float ty = m*tx+b;
			int y = ty;
			temp.at<uchar>(y,x,0) = 127;
			imshow("zzz", temp);
			//waitKey(1);

			if(tSpace.at<uchar>(y,x,0) != fValue){
				cout << "false" << endl;
				return false;
			}
		}
		cout << "return true" << endl;
		return true;
	}
	else{
		if(fPt[1] < sPt[1]){ // set order right
			vector<int> t = sPt;
			sPt = fPt;
			fPt = t;
			cout << "inv" << endl;
		}
		float m = float(fPt[0] - sPt[0]) / float(fPt[1] - sPt[1]);
		float b = float(fPt[0]) - m*float(fPt[1]);

		cout << "y: " << m << " , " << b << endl;

		Mat temp = tSpace;

		for(int x = sPt[1]+1; x<fPt[1]-1; x++){
			int y = round(m*x+b);
			temp.at<uchar>(y,x,0) = 127;
			imshow("zzz", temp);
			//waitKey(1);

			if(tSpace.at<uchar>(x,y,0) != fValue){
				cout << "false" << endl;
				return false;
			}
		}
		cout << "return true" << endl;
		return true;
	}
}

void MiniGraph::getNodeValues(){
	this->nodeValue.erase(this->nodeValue.begin(), this->nodeValue.end());
	for(int i=0; i<this->nmNodes; i++){
		this->nodeValue.push_back(this->nodeReward[i] - this->nodeCost[i]);
	}
}

int MiniGraph::getMaxIndex(vector<float> value){
	int maxdex;
	float maxValue = -INFINITY;
	for(int i=0; i<(int)value.size(); i++){
		if(value[i] > maxValue){
			maxdex = i;
			maxValue  = value[i];
		}
	}
	return maxdex;
}

void MiniGraph::getNodeCosts(int cNode){
	this->nodeCost.erase(this->nodeCost.begin(), this->nodeCost.end());
	for(int i=0; i<(int)this->nmNodes; i++){// for each node
		this->nodeCost.push_back(this->distGraph[cNode][i]);//.push_back(this->aStarDist(cNode,i)); // get A* cost to each Frontier
	}
	cout << "nodeCosts: ";
	for(int i=0;i<this->nmNodes; i++){
		cout << "nodeCost[" << i << "]: " << this->nodeCost[i] << endl;
	}
}

void MiniGraph::getNodeRewards(){
	this->nodeReward.erase(this->nodeReward.begin(), this->nodeReward.end());
	this->nodeFrontiers.erase(this->nodeFrontiers.begin(), this->nodeFrontiers.end());
	for(int i=0; i<this->nmNodes; i++){// for each Frontier
		this->nodeReward.push_back(0);
		vector<int> t;
		this->nodeFrontiers.push_back(t);
	}
	for(int i=0; i<(int)this->frontiers.size(); i++){// for each Frontier
		vector<int> t;
		t.push_back(this->frontiers[i][1]);
		t.push_back(this->frontiers[i][0]);
		int a = findNearestNode(t); // find node closest to Frontier
		this->nodeFrontiers[a].push_back(i);
		this->nodeReward[a] += 50; // sub in froniter value
	}

	for(int i=0;i<this->nmNodes; i++){
		cout << "nodeFrontiers[" << i << "]: ";
		for(int j=0; j<this->nodeFrontiers[i].size(); j++){
			cout << this->nodeFrontiers[i][j] << ", ";
		}
		cout << endl;
	}
}

void MiniGraph::importFrontiers(vector<vector<int> > FrontierList){ // bring in Frontiers to MiniGraph
	this->frontiers.clear();
	this->frontiers = FrontierList;
}

void MiniGraph::importUAVLocations(vector<vector<int> > cLocList){ // bring in UAV locations to MiniGraph
	this->cLocListMap.clear();
	this->cLocListMap = cLocList;
}

int MiniGraph::findNearestNode(vector<int> in){
	float minDist = INFINITY;
	int minIndex;
	for(int i=0; i<this->nmNodes; i++){
		float a = this->euclidianDist(in, this->graf[i]);
		if(a < minDist){
			minDist = a;
			minIndex = i;
		}
	}
	return minIndex;
}

float MiniGraph::aStarDist(int strt, int gl){
	int cSet[this->nmNodes]; // 1 means in closed set, 0 means not
	int oSet[this->nmNodes]; // 1 means in open set, 0 means not
	float gScore[this->nmNodes]; // known cost from initial node to n
	float fScore[this->nmNodes]; // gScore + heuristic score (dist to goal + imposed cost)
	for(int i=0;i<this->nmNodes;i++){
		cSet[i] = 0;
		oSet[i] = 0;
		gScore[i]=INFINITY; // init scores to inf
		fScore[i]=INFINITY; // init scores to inf
	}
	oSet[strt] = 1; // starting node in open set
	gScore[strt] = 0; // starting node has score 0
	fScore[strt] = gScore[strt] + this->euclidianDist(this->graf[strt],this->graf[gl]); // calc score of open set

	int foo = 1;
	int finishFlag = 0;
	while(foo>0 && finishFlag == 0){
		/////////////////// this finds node with lowest fScore and makes current
		float min = INFINITY;
		int iMin = -1;
		for(int i=0;i<this->nmNodes;i++){
			//cerr << "aStar:: "  << i << "/" << oSet[i] << "/" << fScore[i] << "/" << min << endl;
			if(oSet[i] > 0 && fScore[i] < min){
				min = fScore[i];
				iMin = i;
			}
		}
		if(iMin < 0){
			finishFlag = 1;
		}
		else{
			int current = iMin;
			/////////////////////// end finding current node
			if(current == gl){ // if the current node equals goal, then return the distance to the goal
				finishFlag = 1;
				return fScore[gl];
			} ///////////////////////////////// end construct path
			oSet[current] = 0;
			cSet[current] = 1;
			for(int nbr=0;nbr<this->nmNodes;nbr++){
				float tGScore;
				if(this->distGraph[current][nbr] > 0 && this->distGraph[current][nbr] < 1000000){ // find all adj neighbors that are observed
					if(cSet[nbr] == 1){ // has it already been eval? in cSet
						continue;
					}
					tGScore = gScore[current] + this->distGraph[current][nbr]; // calc temporary gscore
					if(oSet[nbr] == 0){
						oSet[nbr] = 1;  // add nbr to open set
					}
					else if(tGScore >= gScore[nbr]){ // is temp gscore better than stored g score of nbr
						continue;
					}
					gScore[nbr] = tGScore;
					fScore[nbr] = gScore[nbr] + this->euclidianDist(this->graf[nbr],this->graf[gl]);
				}
			}
			/////////////// end condition for while loop, check if oSet is empty
			foo = 0;
			for(int i=0;i<this->nmNodes;i++){
				foo+= oSet[i];
			}
		}
	}
	return INFINITY;
}

void MiniGraph::simulateObservation(int node, Mat &viewMat){

	vector<int> cLoc;
	cLoc.push_back(this->graf[node][0]);
	cLoc.push_back(this->graf[node][1]);

	// make perimeter of viewing circle fit on image
	for(size_t i=0; i<this->viewPerim.size(); i++){
		int px = cLoc[0] + this->viewPerim[i][0];
		int py = cLoc[1] + this->viewPerim[i][1];

		bool flag = true;
		while(flag){
			flag = false;
			if(px < 0){
				float dx = cLoc[0] - px;
				float dy = cLoc[1] - py;

				float m = dy / dx;
				float b = cLoc[1]-m*cLoc[0];
				px = 0;
				py = b;
				flag = true;
			}
			else if(px >= viewMat.cols){
				float dx = cLoc[0] - px;
				float dy = cLoc[1] - py;

				float m = dy / dx;
				float b = cLoc[1]-m*cLoc[0];
				px = viewMat.cols-1;
				py = m*px + b;
				flag = true;
			}
			else if(py < 0){
				float dx = cLoc[0] - px;
				float dy = cLoc[1] - py;

				float m = dy / dx;
				float b = cLoc[1]-m*cLoc[0];
				py = 0;
				px = (py-b)/m;
				flag = true;
			}
			else if(py >= viewMat.rows){
				float dx = cLoc[0] - px;
				float dy = cLoc[1] - py;

				float m = dy / dx;
				float b = cLoc[1]-m*cLoc[0];
				py = viewMat.rows-1;
				px = (py-b)/m;
				flag = true;
			}
		}

		// check visibility to all points on circle
		float dx = px - cLoc[0];
		float dy = py - cLoc[1];
		if(dx != 0){
			if(dx > 0){
				float m = dy/dx;
				float b = cLoc[1]-m*cLoc[0];

				int y0 = cLoc[1];
				for(int x0 = cLoc[0]; x0 < px; x0++){
					y0 = m*x0+b;
					if(this->costMap.at<uchar>(x0,y0,0) > 2){
						break;
					}
					else if(this->costMap.at<uchar>(x0,y0,0) == 2){
						viewMat.at<uchar>(x0,y0,0) = 255;
					}
				}
			}
			else{
				float m = dy/dx;
				float b = cLoc[1]-m*cLoc[0];

				int y0 = cLoc[1];
				for(int x0 = cLoc[0]; x0 > px; x0--){
					y0 = m*x0+b;
					if(this->costMap.at<uchar>(x0,y0,0) > 2){
						break;
					}
					else if(this->costMap.at<uchar>(x0,y0,0) == 2){
						viewMat.at<uchar>(x0,y0,0) = 255;
					}
				}
			}
		}
		else{
			if(dy > 0){
				int x0 = cLoc[0];
				for(int y0 = cLoc[1]; y0 < py; y0++){
					if(this->costMap.at<uchar>(x0,y0,0) > 2){
						break;
					}
					else if(this->costMap.at<uchar>(x0,y0,0) == 2){
						viewMat.at<uchar>(x0,y0,0) = 255;
					}
				}
			}
			else{
				int x0 = cLoc[0];
				for(int y0 = cLoc[1]; y0 > py; y0--){
					if(this->costMap.at<uchar>(x0,y0,0) > 2){
						break;
					}
					else if(this->costMap.at<uchar>(x0,y0,0) == 2){
							viewMat.at<uchar>(x0,y0,0) = 255;
					}
				}
			}
		}
	}
}

void MiniGraph::getUnobservedMat(Mat &inputMat){
	Scalar white = 255;
		Scalar gray = 127;

		// 0 = free space
		// 50 = Frontier
		// 100 = unknown
		// Infinity = obstacle

		this->unknownMat = Mat::zeros(inputMat.rows, inputMat.cols,CV_8UC1);

		Mat mapImage = Mat::zeros(inputMat.rows, inputMat.cols,CV_8UC3);
		Vec3b white3; white3[0] = 255; white3[1] = 255; white3[2] = 255;
		Vec3b red; red[0] = 0; red[1] = 0; red[2] = 255;
		Vec3b blue; blue[0] = 255; blue[1] = 0; blue[2] = 0;
		Vec3b gray3; gray3[0] = 127; gray3[1] = 127; gray3[2] = 127;

		for(int i = 0; i<inputMat.cols; i++){
			for(int j =0; j<inputMat.rows; j++){
				Scalar intensity =  inputMat.at<uchar>(i,j,0);
				if(intensity[0] == 100){
					this->unknownMat.at<uchar>(i,j,0) = 255;
					mapImage.at<Vec3b>(i,j) = gray3;
				}
			}
		}

		bitwise_and(this->unknownMat, this->freeMat, this->unknownMat);
		imshow("unknown mat", this->unknownMat);
		waitKey(1);

}

void MiniGraph::createMiniGraph(Mat &inputMat){
	Scalar white = 255;
	Scalar gray = 127;

	// 0 = free space
	// 50 = Frontier
	// 100 = unknown
	// Infinity = obstacle

	this->costMap = inputMat;
	this->obstacleMat = Mat::zeros(inputMat.rows, inputMat.cols,CV_8UC1);
	this->freeMat = Mat::zeros(inputMat.rows, inputMat.cols,CV_8UC1);
	this->rewardMat = Mat::zeros(inputMat.rows, inputMat.cols,CV_8UC1);

	Mat mapImage = Mat::zeros(inputMat.rows, inputMat.cols,CV_8UC3);
	Vec3b white3; white3[0] = 255; white3[1] = 255; white3[2] = 255;
	Vec3b black3; black3[0] = 0; black3[1] = 0; black3[2] = 0;
	Vec3b red; red[0] = 0; red[1] = 0; red[2] = 255;
	Vec3b blue; blue[0] = 255; blue[1] = 0; blue[2] = 0;
	Vec3b gray3; gray3[0] = 127; gray3[1] = 127; gray3[2] = 127;

	// 1 = observed free
	// 2 = inferred free
	// 101 = observed obstacle
	// 102 = inferred obstacle
	// 201 unknown

	for(int i = 0; i<inputMat.cols; i++){
		for(int j =0; j<inputMat.rows; j++){
			Scalar intensity =  inputMat.at<uchar>(i,j,0);
			if(intensity[0] <= 2){ // observed free space or inferred free space - to make travel graph
				this->freeMat.at<uchar>(i,j,0) = 255;
				mapImage.at<Vec3b>(i,j) = white3;
				if(intensity[0] == 2){
					this->rewardMat.at<uchar>(i,j,0) = 255;
				}
			}
			else if(intensity[0] <= 102){ // unknown
				mapImage.at<Vec3b>(i,j) = gray3;
			}
			else if(intensity[0] > 200){ // unknown
				this->obstacleMat.at<uchar>(i,j,0) = 255;
				mapImage.at<Vec3b>(i,j) = black3;
			}
		}
	}

	threshold(this->freeMat,this->freeMat,10,255,CV_THRESH_BINARY);

	// need to apply a mask to this so it only updates areas in view of UAVs
	this->thinning(this->freeMat,this->miniImage);

	vector<vector<int> > MiniGraphNodes;
	for(int i=0; i<this->miniImage.cols; i++){
		for(int j=0; j<this->miniImage.rows; j++){
			Scalar intensity =  this->miniImage.at<uchar>(i,j);
			if(intensity[0] == 255){
				vector<int> t;
				t.push_back(i);
				t.push_back(j);
				//this->graf.push_back(t);
				MiniGraphNodes.push_back(t);
			}
		}
	}

	this->graf.clear();
	// city block distance for graph
	this->findCityBlockDistanceNodes(MiniGraphNodes); // add them to graf ;
	this->cityBlockDistanceNodeConnections(); // find connections in graf;

	//this->breadthFirstSearchNodeConnections(MiniGraphNodes);

	//this->findPointOfInterestNodes();
	//this->condenseGraph();
	//this->breadthFirstSearchAssembleMiniGraph();
	this->cNode = this->findNearestNode(this->cLocList[0]);
}

void MiniGraph::displayCoordMap(){
	Mat coordGraph = Mat::zeros(10*this->miniImage.rows, 10*this->miniImage.cols, CV_8UC3);
	Scalar white;
	white[0] = 255; white[1] = 255; white[2] = 255;
	Scalar gray;
	gray[0] = 127; gray[1] = 127; gray[2] = 127;
	Scalar red;
	red[0] = 0; red[1] = 0; red[2] = 255;
	Scalar green;
	green[0] = 0; green[1] = 255; green[2] = 0;
	Scalar blue;
	blue[0] = 255; blue[1] = 0; blue[2] = 0;

	for(int i=0; i<(int)this->graf.size(); i++){
		Point temp;
		temp.x = this->graf[i][1]*10;
		temp.y = this->graf[i][0]*10;
		circle(coordGraph,temp,1,white,-1,8);
		char str[50];
		sprintf(str,"%d",i);
		putText(coordGraph, str, temp, FONT_HERSHEY_PLAIN,2,green);
	}
	for(int i=0; i<this->nmNodes; i++){
		for(int j=0; j<this->nmNodes; j++){
			if(this->distGraph[i][j] < 1000){
				Point temp, temp2;
				temp.x = this->graf[i][1]*10;
				temp.y = this->graf[i][0]*10;
				temp2.x = this->graf[j][1]*10;
				temp2.y = this->graf[j][0]*10;
				line(coordGraph, temp, temp2, white, 1,8);
			}
		}
	}
	for(size_t i=0;i<this->frontiers.size(); i++){
		Point temp;
		temp.x = this->frontiers[i][1]*10;
		temp.y = this->frontiers[i][0]*10;
		circle(coordGraph,temp,5,red,-1,8);
	}
	for(size_t i=0; i< this->cLocList.size(); i++){
		Point temp;
		temp.x = this->cLocList[i][1]*10;
		temp.y = this->cLocList[i][0]*10;
		circle(coordGraph,temp,10,blue,-1,8);
	}
	Point cn;
	cn.x = this->graf[cNode][1]*10;
	cn.y = this->graf[cNode][0]*10;
	circle(coordGraph,cn,10,blue,2);
	namedWindow("coordGraph", WINDOW_NORMAL);
	imshow("coordGraph", coordGraph);
}

bool MiniGraph::bresenhamLineCheck(vector<int> cLoc, vector<int> cPt){
	float dx = cLoc[0] - cPt[0];
	float dy = cLoc[1] - cPt[1];

	float er = -1;
	float de = 1;
	if(dx != 0){
		de = abs(dy/dx);
	}
	int y = cLoc[1];
	for(int x = cLoc[0]; x<cPt[0]-1; x++){
		if(this->obstacleMat.at<uchar>(x,y,0)){
			return false;
		}
		er = er + de;
		if(er >= 0){
			y++;
			er--;
		}
	}
	return true;

}

void MiniGraph::findCityBlockDistanceNodes(vector<vector<int> > MiniGraphNodes){
	this->graf.clear();
	this->graf.push_back(MiniGraphNodes[0]);
	for(size_t i=1; i<MiniGraphNodes.size(); i++){
		bool tFlag = true;
		for(size_t j=0; j<this->graf.size(); j++){
			if(this->euclidianDist(MiniGraphNodes[i],this->graf[j]) < this->grafSpacing){
				tFlag = false;
			}
		}
		if(tFlag){
			this->graf.push_back(MiniGraphNodes[i]);
		}
	}
	this->nmNodes = this->graf.size();
}

void MiniGraph::cityBlockDistanceNodeConnections(){
	this->distGraph.clear();
	for(int i=0; i<this->nmNodes; i++){
		vector<float> asdf;
		for(int j=0; j<this->nmNodes; j++){
			asdf.push_back(INFINITY);
		}
		this->distGraph.push_back(asdf);
	}

	for(int i=0; i<this->nmNodes; i++){
		for(int j=0; j<this->nmNodes; j++){
			if(this->euclidianDist(graf[i],graf[j]) < this->grafConnectionSpacing && i != j){
				this->distGraph[i][j] = this->euclidianDist(graf[i],graf[j]);
			}
		}
	}
}

void MiniGraph::breadthFirstSearchNodeConnections(vector<vector<int> > openNodes){

	this->graf.clear();
	this->graf.push_back(openNodes[0]);
	cerr << "0" << endl;
	while(openNodes.size() > 0){

		int mindex = 0;
		float mindist = INFINITY;

		size_t i =0;
		cerr << "on.size(): " << openNodes.size() << endl;
		while(i < openNodes.size()){ // while there are open nodes left to check
			bool flag = true;
			int myMindex = -1;
			float myMindist = INFINITY;

			for(size_t j=0; j<this->graf.size(); j++){ // check against all nodes already in graf
				float dist = this->euclidianDist(openNodes[i], this->graf[j]);
				cerr << "   mindist / myMinDist , dist: " << myMindist << " , " << dist;
				if(dist < this->grafSpacing){
					cerr << ": node removed" << endl;
					openNodes.erase(openNodes.begin()+i);
					flag = false;
					break;
				}
				else if(dist < mindist){
					cerr << ": new closest node" << endl;
					myMindist = dist;
					myMindex = j;
				}
				cerr << "end for loop: " << myMindex << endl;
			}
			if(flag && myMindist < mindist){
				mindist = myMindist;
				mindex = i;
			}
			i++;
		}
		cerr << "mindex / on.size(): " << mindex << " / " << openNodes.size() << endl;
		if(openNodes.size() > 0){
			this->graf.push_back(openNodes[mindex]);
		}
		cerr << "4" << endl;
	}
}

void MiniGraph::cornerFinder(Mat &inputMat){
	this->corners.clear();
	for(size_t i=1; i<inputMat.rows-1; i++){
		for(size_t j=1; j<inputMat.cols-1; j++){
			int xp = inputMat.at<uchar>(i+1,j);
			int xm = inputMat.at<uchar>(i-1,j);

			if(xp != xm){
				int yp = inputMat.at<uchar>(i,j+1);
				int ym = inputMat.at<uchar>(i,j-1);

				if(yp != ym){
					vector<int> c;
					c.push_back(i);
					c.push_back(j);
					this->corners.push_back(c);
				}
			}
		}
	}
}

bool MiniGraph::bisectionCheck(vector<int> a, vector<int> b){
	if(this->cityBlockDist(a,b) > 2){ // do I bisect further?
		vector<int> c;
		c.push_back((a[0] + b[0])/2); // find midpoint
		c.push_back((a[1] + b[1])/2);
		if(this->miniImage.at<uchar>(c[0],c[1],0) == 0){ // is midpoint an obstacle?
			return false;
		}
		else{ // midpoint is not an obstacle
			if(this->bisectionCheck(a,c) && this->bisectionCheck(b,c)){
				return true;
			}
		}
	}
	else{ // end of bisection
		return true;
	}
}

float MiniGraph::cityBlockDist(vector<int> a, vector<int> b){
	float d = abs(a[0] - b[0]) + abs(a[1]+b[1]);
	return d;
}

float MiniGraph::euclidianDist(vector<int> a, vector<int> b){
	float d = sqrt(pow(a[0]-b[0],2) + pow(a[1] - b[1],2));
	return d;
}

void MiniGraph::thinning(const Mat& src, Mat& dst){
	//https://github.com/bsdnoobz/zhang-suen-thinning/blob/master/thinning.cpp
    dst = src.clone();
    dst /= 255;         // convert to binary image

    cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningIteration(dst, 0);
        thinningIteration(dst, 1);
        cv::absdiff(dst, prev, diff);
        dst.copyTo(prev);
    }
    while (cv::countNonZero(diff) > 0);

    dst *= 255;
}

void MiniGraph::thinningIteration(Mat& img, int iter){
	//https://github.com/bsdnoobz/zhang-suen-thinning/blob/master/thinning.cpp
    CV_Assert(img.channels() == 1);
    CV_Assert(img.depth() != sizeof(uchar));
    CV_Assert(img.rows > 3 && img.cols > 3);

    cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

    int x, y;
    uchar *pAbove;
    uchar *pCurr;
    uchar *pBelow;
    uchar *nw, *no, *ne;    // north (pAbove)
    uchar *we, *me, *ea;
    uchar *sw, *so, *se;    // south (pBelow)

    uchar *pDst;

    // initialize row pointers
    pAbove = NULL;
    pCurr  = img.ptr<uchar>(0);
    pBelow = img.ptr<uchar>(1);

    for (y = 1; y < img.rows-1; ++y) {
        // shift the rows up by one
        pAbove = pCurr;
        pCurr  = pBelow;
        pBelow = img.ptr<uchar>(y+1);

        pDst = marker.ptr<uchar>(y);

        // initialize col pointers
        no = &(pAbove[0]);
        ne = &(pAbove[1]);
        me = &(pCurr[0]);
        ea = &(pCurr[1]);
        so = &(pBelow[0]);
        se = &(pBelow[1]);

        for (x = 1; x < img.cols-1; ++x) {
            // shift col pointers left by one (scan left to right)
            nw = no;
            no = ne;
            ne = &(pAbove[x+1]);
            we = me;
            me = ea;
            ea = &(pCurr[x+1]);
            sw = so;
            so = se;
            se = &(pBelow[x+1]);

            int A  = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
                     (*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
                     (*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
                     (*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
            int B  = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
            int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
            int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0){
                pDst[x] = 1;
            }
        }
    }

    img &= ~marker;
}

MiniGraph::~MiniGraph() {
	// TODO Auto-generated destructor stub
}

void MiniGraph::breadthFirstSearchAssembleMiniGraph(){
	// breadth first search a random node to all other nodes to get travel distances
	// during search check if each location is a node and log distance then remove that branch from the search
	// start search again from next nodes, think of this as exploring all edges attached to one node to get dist;

	// create a list of nodes
	vector<vector<int> > nLoc;
	for(int i=0; i<this->nmNodes; i++){
		vector<int> tempList;
		tempList.push_back(this->graf[i][0]);
		tempList.push_back(this->graf[i][1]);
		//cout << this->graf[i].x << " & " << this->graf[i].y << endl;
		nLoc.push_back(tempList);
	}

	// init everything
	// init distgraph for distances between all nodes
	this->distGraph.erase(this->distGraph.begin(),this->distGraph.end());
	for(int i=0; i<this->nmNodes; i++){
		vector<float> asdf;
		for(int j=0; j<this->nmNodes; j++){
			asdf.push_back(INFINITY);
		}
		this->distGraph.push_back(asdf);
	}

	int flag[this->miniImage.rows][this->miniImage.cols]; // is this node traversable
	for(int i=0; i<this->miniImage.rows; i++){
		vector<float> asdf;
		for(int j=0; j<this->miniImage.cols; j++){
			asdf.push_back(0);
			flag[i][j] = (this->miniImage.at<uchar>(j,i) !=0); // is it traverseable? 1-yes, 0-no
		}
		this->distGraph.push_back(asdf);
	}

	//for(int i=0; i<this->miniImage.rows; i++){
	//	for(int j=0; j<this->miniImage.cols; j++){
	//		for(int k=0; k<nLoc.size(); k++){
	//			if(i==nLoc[k][0] && j==nLoc[k][1]){
	//				cout << "*";
	//			}
	//			else{
	//				cout << flag[i][j];
	//			}
	//		}
	//	}
	//	cout << endl;
	//}


	//cout << "There are " << this->nmNodes << endl;
	//Mat coordGraph(10*this->miniImage.rows, 10*this->miniImage.cols, CV_8UC1);
	for(int seed = 0; seed<(int)this->nmNodes; seed++){
		vector<vector<int> > oSet; // stores locations
		vector<float> dist; // stores distance to all members in oSet
		vector<vector<int> > cSet; // stores locations of members in closed set
		vector<vector<int> > foundNode; // stores location of found nodes

		// initialize open set
		vector<int> o;
		o.push_back(nLoc[seed][0]);
		o.push_back(nLoc[seed][1]);
		oSet.push_back(o); //

		// initialize distance to first item in open set
		dist.push_back(0); // starting location has a distance of 0

		//cout << "BEGIN ITERATION WITH SEED " << seed << " /////// nLoc: " << nLoc[seed][0] << "< " << nLoc[seed][1] << endl;
		while((int)oSet.size() > 0){
			//cout << "   oSet: ";
			//for(int k=0; k<(int)oSet.size(); k++){
			//	cout << oSet[k][0] << "," << oSet[k][1] << "; ";
			//}
			//cout << endl;
			// find closestNode to seed in the open set; i.e. the minimum distance
			float minDist = INFINITY;
			int minLoc[2];
			int mindex;
			for(int i=0; i<(int)oSet.size(); i++){
				if(dist[i] < minDist){
					minDist = dist[i];
					mindex = i;
					minLoc[0]= oSet[i][0];
					minLoc[1] = oSet[i][1];
				}
			}
			//cout << "minDex: " << minLoc[0] << "," << minLoc[1] << endl;

			// is mindex at an undiscovered node?
			//cout << "checking if minDex is at an undiscovered Node" << endl;
			for(int j = -2; j<3; j++){
				for(int k=-2; k<3; k++){

					for(int i=0; i<this->nmNodes; i++){ // am I at a new node?
						if(nLoc[i][0] == minLoc[0]+j && nLoc[i][1] == minLoc[1]+k && i != seed){
				//			cout << "  Found node " << i << " from seed " << seed << " with a dist of " << dist[mindex]+sqrt(pow(j,2)+pow(k,2)) << endl;
							this->distGraph[seed][i] = dist[mindex]+sqrt(pow(j,2)+pow(k,2));
							this->distGraph[i][seed] = dist[mindex]+sqrt(pow(j,2)+pow(k,2));
							vector<int> t;
							t.push_back(nLoc[i][0]);
							t.push_back(nLoc[i][1]);
							foundNode.push_back(t);
						}
					}
				}
			}

			// is mindex near a discovered node? if so, don't expand
			//cout << "checking if minDex is near a discovered node" << endl;
			//for(int i=0; i<(int)foundNode.size(); i++){
			//	if(sqrt(pow(foundNode[i][0]-minLoc[0] && foundNode[i][1] == minLoc[1],2)) < 1){ // am I near a node I have found before?
			//		cout << "   near a discovered node" << endl;
			//		skip = true;
			//	}
			//}

			// check minDex's nbrs to see if they should be added to the open set
			//cout << "checking nbrs" << endl;

			//for(int i = -2; i<3; i++){
			//	for(int j=-2; j<3; j++){
			//		if(i == 0 && j == 0){
			//			cout << "*";
			//		}
			//		else{
			//			cout << flag[minLoc[0] + i][minLoc[1] + j];
			//		}
			//	}
			//	cout << endl;
			//}

			for(int i = -3; i<4; i++){
				for(int j=-3; j<4; j++){
					if(flag[minLoc[0] + i][minLoc[1] + j] == 1){ // traversable
						bool cFlag = true;
						for(int k=0; k<(int)cSet.size(); k++){ // not in closed set
							if(cSet[k][0] == minLoc[0] + i && cSet[k][1] == minLoc[1] + j){
								cFlag = false;
							}
						}
						for(int k=0; k<(int)oSet.size(); k++){ // not in open set
							if(oSet[k][0] == minLoc[0] + i && oSet[k][1] == minLoc[1] + j){
								cFlag = false;
							}
						}
						if(cFlag){ // add to openSet
							vector<int> o;
							o.push_back(minLoc[0] + i);
							o.push_back(minLoc[1] + j);
							oSet.push_back(o);

							dist.push_back(dist[mindex] + sqrt(pow(i,2)+pow(j,2))); // get distance
							//cout << "   found a nbr: " << minLoc[0] + i << "," << minLoc[1] + j << " at dist: " <<  dist[mindex] + sqrt(pow(i,2)+pow(j,2)) << endl;
						}
					}
				}
				//cout << "out" << endl;
			}
			// move the current node out of open set and into closed set
			//cout << "moving minDex to closed set" << endl;
			vector<int> ml;
			ml.push_back(minLoc[0]);
			ml.push_back(minLoc[1]);
			cSet.push_back(ml);

			//cout << "   cSet: ";
			//for(int k=0; k<(int)cSet.size(); k++){
			//	cout << cSet[k][0] << "," << cSet[k][1] << "; ";
			//}
			//cout << endl;

			//cout << "   oSet: ";
			//for(int k=0; k<(int)oSet.size(); k++){
			//	cout << oSet[k][0] << "," << oSet[k][1] << "; ";
			//}
			//cout << endl;

			oSet.erase(oSet.begin()+mindex,oSet.begin()+mindex+1);
			dist.erase(dist.begin()+mindex,dist.begin()+mindex+1);
		}
	}

	for(int i=0; i<this->nmNodes; i++){
		this->distGraph[i][i] = 0;
	}

	//cout << "DISTGRAPH:" << endl;
	//for(int i=0; i<this->nmNodes; i++){
	//	for(int j=0; j<this->nmNodes; j++){
	//		cout << floor(100*this->distGraph[i][j])/100 << " , ";
	//	}
	//	cout << endl;
	//}

	/*
	Mat coordGraph(10*this->miniImage.rows, 10*this->miniImage.cols, CV_8UC1);
	for(int i=0; i<(int)this->graf.size(); i++){
		Point temp;
		temp.x = this->graf[i][0]*10;
		temp.y = this->graf[i][1]*10;
		circle(coordGraph,temp,2,white,-1,8);
		char str[50];
		sprintf(str,"%d",i);
		putText(coordGraph, str, temp, FONT_HERSHEY_PLAIN,2,white);
	}
	for(int i=0; i<this->nmNodes; i++){
		for(int j=0; j<this->nmNodes; j++){
			if(this->distGraph[i][j] < 1000){
				Point temp, temp2;
				temp.x = this->graf[i][0]*10;
				temp.y = this->graf[i][1]*10;
				temp2.x = this->graf[j][0]*10;
				temp2.y = this->graf[j][1]*10;
				line(coordGraph, temp, temp2, white, 1,8);
			}
		}
	}
	imshow("coordGraph", coordGraph);
	waitKey(1);
	*/
}

void MiniGraph::condenseGraph(){
	vector<vector<int> > keep;
	for(size_t i=0; i<this->graf.size(); i++){
		bool flag = true;
		for(size_t j=0; j<this->graf.size(); j++){
			if(this->euclidianDist(this->graf[i], this->graf[j]) < this->grafConnectionSpacing && i != j){

				int a = (this->graf[i][0] + this->graf[j][0])/2;
				int b = (this->graf[i][1] + this->graf[j][1])/2;
				vector<int> c;
				c.push_back(a);
				c.push_back(b);
				keep.push_back(c);
				flag = false;
			}
		}
		if(flag){
			keep.push_back(this->graf[i]);
		}
	}
	this->graf.clear();

	this->graf = keep;
}


void MiniGraph::findPointOfInterestNodes(){

	int x, y;

	uchar *pAbove;
	uchar *pCurr;
	uchar *pBelow;

	uchar *bb, *bc, *bd;
	uchar *cb, *cc, *cd;
	uchar *db, *dc, *dd;

	// initialize row pointers
	pAbove = NULL;
	pCurr  = this->miniImage.ptr<uchar>(0);
	pBelow = this->miniImage.ptr<uchar>(1);

	this->graf.clear();
	for (y = 1; y < this->miniImage.rows-1; ++y) {
		// shift the rows up by one
		pAbove = pCurr;
		pCurr  = pBelow;
		pBelow = this->miniImage.ptr<uchar>(y+1);

		// initialize col pointers

		bb = &(pAbove[0]);
		bc = &(pAbove[1]);
		bd = &(pAbove[2]);

		cb = &(pCurr[0]);
		cc = &(pCurr[1]);
		cd = &(pCurr[2]);

		db = &(pBelow[0]);
		dc = &(pBelow[1]);
		dd = &(pBelow[2]);

		for (x = 1; x < this->miniImage.cols-1; ++x) {
			// shift col pointers left by one (scan left to right)
			bb = bc;
			bc = bd;
			bd = &(pAbove[x+2]);

			cb = cc;
			cc = cd;
			cd = &(pCurr[x+2]);

			db = dc;
			dc = dd;
			dd = &(pBelow[x+1]);

			int outerEdge[9];
			outerEdge[0] = (*bb != 0);
			outerEdge[1] = (*bc != 0);
			outerEdge[2] = (*bd != 0);
			outerEdge[3] = (*cd != 0);
			outerEdge[4] = (*dd != 0);
			outerEdge[5] = (*dc != 0);
			outerEdge[6] = (*db != 0);
			outerEdge[7] = (*cb != 0);
			outerEdge[8] = (*bb != 0);

			int edgeDetector = 0;
			for(int i=0; i<8; i++){
				if(outerEdge[i] != outerEdge[i+1]){
					edgeDetector++;
				}
			}

			if(*cc != 0 && edgeDetector != 4 && edgeDetector != 0){ // is the center pixel traversable && if 4 edges then there is one way into and one way out of the traversable path, not a node && is there a way to the node
				vector<int> t;
				t.push_back(x);
				t.push_back(y);
				this->graf.push_back(t);
				cout << "vec: ";
				for(int i=0; i<9; i++){
					cout << outerEdge[i] << ",";
				}

			    cout << endl;
			    cout << "edgeDetector: " << edgeDetector << endl;
			    cout << (*bb != 0) << "," << (*bc != 0) << "," << (*bd != 0) << endl;
			    cout << (*cb != 0) << "," << (*cc != 0) << "," << (*cd != 0) << endl;
			    cout << (*db != 0) << "," << (*dc != 0) << "," << (*dd != 0) << endl << endl;
			}
		}
	}
}


void MiniGraph::invertImageAroundPt(Mat &src, Mat &dst, vector<int> cLoc){

	imshow( "src", src);
	waitKey(1);

	vector<vector<int> > corr;
	float invDist = 20;
	// check a pixel and verify it needs to be inverted
	dst =Mat::zeros(src.cols, src.rows, CV_8UC1);
	for(int i =0; i<src.cols; i++){
		for(int j=0; j<src.rows; j++){ // go through complete source image
			if(src.at<uchar>(i,j,0) > 0){ // do we care about this point?
				float l = sqrt(pow(cLoc[0]-i,2) + pow(cLoc[1]-j,2)); // distance to point
				float t = atan2((cLoc[1]-j) , (cLoc[0]-i)); // angle to point
;
				float nl = invDist / l;
				float x = cLoc[0] + nl * cos(t);
				float y = cLoc[1] + nl * sin(t);
				dst.at<uchar> (int(round(x)),int(round(y)),0) = 255;
				vector<int> c;
				c.push_back(x);c.push_back(y);c.push_back(i);c.push_back(j);
				corr.push_back(c); // keep track of corresponding points
			}
		}
	}

	imshow( "dst", dst);
	waitKey(1);

	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Find contours
	int thresh = 100;
	int max_thresh = 255;
	Canny( dst, canny_output, thresh, thresh*2, 3 );
	findContours( canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	/// Draw contours
	Mat drawing = Mat::zeros( canny_output.size(), CV_8UC1 );
	for( int i = 0; i< contours.size(); i++ ){
		Scalar color = 255;
	    drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
	}

	// need to find the outer contour and then pull all points from it, find them in the corr list to get their original position to include them in the original frame

	/// Show in a window
	imshow( "Contours", drawing );
	waitKey(1);
}


Mat MiniGraph::breadthFirstSearchFindRoom(Mat &src, vector<int> pt){

	// find the clostest point to the identified point
	float minDist = INFINITY;
	vector<int> cPt;
	cPt.push_back(-1);
	cPt.push_back(-1);
	for(int i =0; i<src.cols; i++){
		for(int j=0; j<src.rows; j++){ // go through complete source image
			if(src.at<uchar>(i,j,0) > 0){ // do we care about this point?
				float d = pow(pt[0]-i,2) + pow(pt[1]-j,2);
				if(d < minDist){
					minDist = d;
					cPt[0] = i;
					cPt[1] = j;
				}
			}
		}
	}

	//find adjacent points to the closest point and add to the openSet
	vector<vector<int> > cSet;
	vector<vector<int> > oSet;
	for(int i=-1; i<2; i++){
		for(int j=-1; j<2; j++){
			if(i !=0 || j !=0){
				if(src.at<uchar>(i+cPt[0],j+cPt[1],0) > 0){
					vector<int> t;
					t.push_back(i+cPt[0]);
					t.push_back(j+cPt[1]);
					oSet.push_back(t);
				}
				cout << endl;
			}
		}
	}

	// add initial point to the closedSet
	cSet.push_back(cPt);

	// while there are still points in the openSet
	while(oSet.size() > 0){
		// add current point to closedSet and remove from openSet
		cPt = oSet[oSet.size()-1];
		cout << cPt[0] << " < " << cPt[1] << endl;
		cSet.push_back(cPt);
		oSet.pop_back();

		// find all adjacent points to cPt
		vector<vector<int> > temp;
		for(int i=-1; i<2; i++){
			for(int j=-1; j<2; j++){
				if(i !=0 || j !=0){
					if(src.at<uchar>(i+cPt[0],j+cPt[1],0) > 0){
						vector<int> t;
						t.push_back(i+cPt[0]);
						t.push_back(j+cPt[1]);

						bool flag = true;
						for(size_t k=0; k<cSet.size(); k++){
							if(t == cSet[k]){
								flag = false;
								break;
							}
						}
						if(flag){
							temp.push_back(t);
						}
					}
				}
			}
		}
		// if there is more than 1 adjacent point, add closest to oSet
		if(temp.size() > 1){
			float minDist = INFINITY;
			float mindex = -1;
			for(size_t i=0; i<temp.size(); i++){
				float d = pow(pt[0]-temp[i][0],2) + pow(pt[1]-temp[i][1],2);
				if(d < minDist){
					minDist = d;
					mindex = i;

				}
			}
			oSet.push_back(temp[mindex]);
		}
		else if(temp.size() > 0){
			oSet.push_back(temp[0]);
		}
	}

	Mat dst =Mat::zeros(src.cols, src.rows, CV_8UC1);

	for(size_t i=0; i<cSet.size(); i++){
		src.at<uchar>(cSet[i][0], cSet[i][1],0) = 255;
	}

	imshow("src", src);
	waitKey(1);

	return dst;
}


void MiniGraph::growObstacles(){
	// identify obstacles of length > x
	// determine obstacle orientation and curvature
	// extend obstacle per both
}


void MiniGraph::extractInferenceContour(){
	Mat temp;

	bitwise_or(this->obstacleMat, this->inferenceMat, temp);
	bitwise_or(temp, this->freeMat, temp);
	bitwise_not(temp,temp);

	namedWindow("Inference Contours", WINDOW_AUTOSIZE);
	imshow("Inference Contours", temp);
}


void MiniGraph::growFrontiers(vector<Frontier> frnt){

	// find all Frontiers with traversable bath through free and observed space between them using line check

	// find all members of each Frontier and Frontier orientation^-1 and add all to oSet
	// each member of open set extend one cell in direction orient^-1 and add current cell to closed set and extended cell to oset
	// repeat

	/* Frontier useful class members
	vector<float> orient; // unit vector descirbing orientation
	vector<int> centroid; // [x/y]
	vector<vector<int> > members; // [list][x/y]
	*/

	// get Mat of obstacles and inference combined
	Mat temp;
	bitwise_or(this->obstacleMat, this->inferenceMat, temp);

	threshold(temp,temp,10,255,CV_THRESH_BINARY);
	for(size_t i=0; i<frnt.size(); i++){
		temp.at<uchar>(frnt[i].centroid[0],frnt[i].centroid[1],0) = 255;
	}

	// find all Frontiers with traversable bath through free and observed space between them using line check
	vector<vector<Frontier> > mFrnts;
	for(size_t i=0; i<frnt.size()-1; i++){
		vector<Frontier> t;
		t.push_back(frnt[i]);
		for(size_t j=i+1; j<frnt.size(); j++){
			if(this->lineTraversabilityCheck(this->freeMat, frnt[i].centroid, frnt[j].centroid, 255)){
				t.push_back(frnt[j]);
				line(temp, Point{frnt[i].centroid[1], frnt[i].centroid[0]},Point{frnt[j].centroid[1], frnt[j].centroid[0]}, Scalar(127), 1, 8);
			}
		}
		mFrnts.push_back(t);
	}

	imshow("temp obs", temp);
	waitKey(1);

	imshow("explored", this->freeMat);
	waitKey(1);


	/*
	// all points in Frontier extend in direction of orient one unit as long as it is free space
	for(size_t i=0; i<frnt.members.size(); i++){
		src.at<uchar>(frnt.members[i][0], frnt.members[i][1],0) = 255;
	}

	imshow("src", src);
	waitKey(1);

	// check visibility to all points on circle
	float dx = frnt.orient[0];
	float dy = frnt.orient[1];

	if(dx != 0){
		if(dx > 0){
			float m = dy/dx;
			float b = cLoc[1]-m*cLoc[0];

			int y0 = cLoc[1];
			for(int x0 = cLoc[0]; x0 < px; x0++){
				y0 = m*x0+b;
				if(this->obstacleMat.at<uchar>(x0,y0,0) > 0){
					break;
				}
				else{
					viewMat.at<uchar>(x0,y0,0) = 255;
				}
			}
		}
		else{
			float m = dy/dx;
			float b = cLoc[1]-m*cLoc[0];

			int y0 = cLoc[1];
			for(int x0 = cLoc[0]; x0 > px; x0--){
				y0 = m*x0+b;
				if(this->obstacleMat.at<uchar>(x0,y0,0) > 0){
					break;
				}
				else{
					viewMat.at<uchar>(x0,y0,0) = 255;
				}
			}
		}
	}
	else{
		if(dy > 0){
			int x0 = cLoc[0];
			for(int y0 = cLoc[1]; y0 < py; y0++){
				if(this->obstacleMat.at<uchar>(x0,y0,0) > 0){
					break;
				}
				else{
					viewMat.at<uchar>(x0,y0,0) = 255;
				}
			}
		}
		else{
			int x0 = cLoc[0];
			for(int y0 = cLoc[1]; y0 > py; y0--){
				if(this->obstacleMat.at<uchar>(x0,y0,0) > 0){
					break;
				}
				else{
					viewMat.at<uchar>(x0,y0,0) = 255;
				}
			}

		}
	}

*/
}

/*
float MiniGraph::aStarDist(int strt, int gl){
	vector<int> cSet; // 1 means in closed set, 0 means not
	vector<int> oSet; // 1 means in open set, 0 means not
	vector<float> gScore; // known cost from initial node to n
	vector<float> fScore; // gScore + heuristic score (dist to goal + imposed cost)
	vector<int> cameFrom; // each square has a vector of the location it came from
	for(int i=0;i<this->nmNodes; i++){
		cSet.push_back(0);
		oSet.push_back(0);
		cameFrom.push_back(0);
		gScore.push_back(INFINITY); // init scores to inf
		fScore.push_back(INFINITY); // init scores to inf
	}
	oSet[strt] = 1; // starting node has score 0
	gScore[strt] = 0; // starting node in open set
	fScore[strt] = gScore[strt] + this->distGraph[strt][gl];
	int foo = 1;
	int finishFlag = 0;
	while(foo>0 && finishFlag == 0){
		/////////////////// this finds node with lowest fScore and makes current
		float min = INFINITY;
		int iMin = 0;
		for(int i=0; i<this->nmNodes; i++){
			if(oSet[i] > 0 && fScore[i] < min){
				min = fScore[i];
				iMin = i;
			}
		}
		int cLoc = iMin;
		/////////////////////// end finding current node
		if(cLoc == gl){ // if the current node equals goal, construct path
			finishFlag = 1;
			return gScore[gl];
		} ///////////////////////////////// end construct path
		oSet[cLoc] = 0;
		cSet[cLoc] = 1;
		for(int nbr=0; nbr<this->nmNodes;nbr++){
			if(this->distGraph[cLoc][nbr] < 3){
				float tGScore;
				if(cSet[nbr] == 1){ // has it already been eval? in cSet
					continue;
				}
				tGScore = gScore[cLoc] + this->distGraph[nbr][gl]; // calc temporary gscore
				if(oSet[nbr] == 0){
					oSet[nbr] = 1;  // add nbr to open set
				}
				else if(tGScore >= gScore[nbr]){ // is temp gscore better than stored g score of nbr
					continue;
				}
				cameFrom[nbr] = cLoc;
				gScore[nbr] = tGScore;
				fScore[nbr] = gScore[nbr] + this->distGraph[gl][nbr];
			}
		}
		/////////////// end condition for while loop, check if oSet is empty
		foo = 0;
		for(int i=0; i<this->nmNodes; i++){
			foo+= oSet[i];
		}
	}
	return INFINITY;
}
*/

/*
void MiniGraph::buildTree(){

	int maxPulls = 2;

	Mat myVisibleMat = Mat::zeros(this->miniImage.rows, this->miniImage.cols,CV_8UC1);
	vector<int> t;
	t.push_back(this->graf[this->cNode][0]);
	t.push_back(this->graf[this->cNode][1]);

	this->observe(t,myVisibleMat);
	treeNode myTree(true, this->distGraph, this->cNode, myVisibleMat);
	myTree.value = this->matReward(myVisibleMat);
	while(myTree.nPulls < maxPulls){
		myTree.searchTree(myTree.value);
	}
	cout << "myState: " << myTree.myState << endl;
	cout << "children.size(): " << myTree.children.size() << endl;
	for(size_t i=0; i < myTree.children.size(); i++){
		cout << "   " << myTree.children[i].myState << endl;
	}
	waitKey(1);
}
*/


