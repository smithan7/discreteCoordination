/*
 * TreeNode.cpp
 *
 *  Created on: May 18, 2016
 *      Author: andy
 */

#include "TreeNode.h"

TreeNode::TreeNode(int state, vector<vector<float> > &distGraph, vector<Mat> &observations, vector<int> inPath, int depth, float pathLength){
	this->nPulls = 0;
	this->myState = state;
	this->myPath = inPath;
	this->myPath.push_back(this->myState);
	this->myDepth = depth + 1;
	this->searchComplete = false;
	this->pathLength = pathLength;

	//cerr << "TreeNode::into GetNodeReward" << endl;
	this->getNodeReward(observations);
	//cerr << "TreeNode::out of getNodeReward: " << this->reward << endl;
}

void TreeNode::searchTree(vector<vector<float> > &distGraph, vector<Mat> &observations, float maxLength){
	//cerr << "searchTree::in: " << this->pathLength << endl;
	if(this->pathLength < maxLength){
		//cerr << "searchTree::1" << endl;
		this->nPulls++;
		if(this->nPulls > 1){ // am I getting tried for the 1st time?
			//cerr << "searchTree::nth pull, going into ucb" << endl;
			int s = this->UCBChildSelect();
			//cerr << "searchTree::nth pull, out of ucb, into search tree" << endl;
			//int s = this->eGreedyChildSelect(0.8);
			if(s >= 0){
				this->children[s].searchTree(distGraph, observations, maxLength);
				//cerr << "searchTree::nth pull, searched tree, into update reward" << endl;
				this->updateMyReward();
				//cerr << "searchTree::nth pull, updated reward" << endl;
				//this->updateMyValue(passedValue);
			}
		}
		else{ // my 1st time!
			//cerr << "searchTree::first search, getting children" << endl;
			this->getChildren(distGraph, observations);
			//cerr << "searchTree::got children, updating reward" << endl;
			this->updateMyReward();
			//cerr << "searchTree::got children, updated reward: " << this->reward << endl;
		}
	}
	else{
		this->searchComplete = true;
	}
}

void TreeNode::getNodeReward(vector<Mat> &observations){
	Mat pathMat = Mat::zeros(observations[0].size(), CV_8UC1);
	for(size_t i=0; i<this->myPath.size()-1; i++){
		bitwise_or(pathMat,observations[myPath[i]], pathMat);
	}
	float rewardCntr = 0;
	for(int i=0; i<pathMat.cols; i++){
		for(int j=0; j<pathMat.rows; j++){
			if(pathMat.at<uchar>(i,j,0) == 255){
				rewardCntr++;
			}
		}
	}
	this->reward = rewardCntr * pow(0.99, float(this->myDepth) );
}

void TreeNode::getChildren(vector<vector<float> > &distGraph, vector<Mat> &observations){
	for(size_t i=0; i<distGraph[this->myState].size(); i++){
		float dist = distGraph[this->myState][i];
		if(dist != INFINITY){
			TreeNode a(int(i), distGraph, observations, this->myPath, this->myDepth, this->pathLength+dist);
			this->children.push_back(a);
		}
	}
}

void TreeNode::exploitTree(vector<int>& myPath, vector<float> &rewards){
	if(this->myDepth == 0){
		myPath.clear();
	}
	myPath.push_back(this->myState);
	rewards.push_back(this->reward);
	if(this->children.size() > 0){
		int s = this->greedyChildSelect();
		this->children[s].exploitTree(myPath, rewards);
	}
}

void TreeNode::updateMyReward(){
	float av = 0;
	for(size_t i=0; i<this->children.size(); i++){
		av += this->children[i].reward;
	}
	this->reward = av / this->children.size();
}


int TreeNode::UCBChildSelect(){
	int s = -1; // argmax_(each child) [value(each child) + sqrt(2*ln n(all children) / n(each child]
	float v = -1;
	if(this->children.size() > 0){
		bool flag = true;
		for(size_t i=0; i<this->children.size(); i++){
			if(!this->children[i].searchComplete){
				flag = false;
				float cv = children[i].reward;
				float nPV = 500*sqrt( log(float(this->nPulls)) / float(children[i].nPulls));
				//cerr << "cv < nPV: " << cv << " < " << nPV << endl;
				float tv = cv + nPV;
				if(tv > v){
					v = tv;
					s = i;
				}
			}
		}
		if(flag){
			this->searchComplete = true;
		}
	}
	return s;
}

int TreeNode::greedyChildSelect(){
	int s = -1;
	float v = -1;
	for(size_t i=0; i<this->children.size(); i++){
		float tv = float(children[i].reward);
		if(tv > v){
			v = tv;
			s = i;
		}
	}
	return s;
}

int TreeNode::eGreedyChildSelect(float epsilon = 0.5){
	int s = -1;
	if( (rand() % 10000)/10000 > epsilon){
		float v = -1;
		for(size_t i=0; i<this->children.size(); i++){
			float tv = children[i].reward;
			if(tv > v){
				v = tv;
				s = i;
			}
		}
	}
	else{
		s = (rand() % children.size()) + 1;
	}
	return s;
}

int TreeNode::randChildSelect(){
	int s = (rand() % children.size()) + 1;
	return s;
}


int TreeNode::simAnnealingChildSelect(float& temp, float cooling = 0.99){
	if(this->myDepth == 0){
		temp = temp * cooling;
	}
	int s = -1;
	float v = -1;
	vector<float> vc;
	for(size_t i=0; i<this->children.size(); i++){
		vc.push_back(children[i].reward);
		if(vc[i] > v){
			v = vc[i];
			s = i;
		}
	}
	vector<float> p;
	for(size_t i=0; i<vc.size(); i++){
		p.push_back(exp(-(v-vc[i])/temp)); // do sim annealing probability
	}
	return s;
}


TreeNode::~TreeNode() {

}
