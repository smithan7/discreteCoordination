/*
 * treeNode.cpp
 *
 *  Created on: May 18, 2016
 *      Author: andy
 */

#include "treeNode.h"

treeNode::treeNode(int state, miniGraph &miniMaster, vector<int> inPath, int depth){
	cerr << "into make tree node" << endl;
	this->value = 0;
	this->nPulls = 0;
	this->myState = state;
	this->myPath = inPath;
	this->myPath.push_back(this->myState);
	this->myDepth = depth + 1;
	this->searchComplete = false;

	cerr << "into evaluate node" << endl;
	this->evaluateNode(miniMaster);
	cerr << "out of evaluate node" << endl;
}

void treeNode::searchTree(miniGraph &miniMaster){
	if(!this->searchComplete){
		cerr << "into search tree" << endl;
		Mat temp = miniMaster.miniImage.clone();
		cerr << "2" << endl;
		cerr << "nPulls: " << this->nPulls << endl;
		cerr << "myPath.size(): " << myPath.size() << endl;
		if(this->myPath.size() > 0 && this->myPath.size() < 100){
			cerr << "2.1" << endl;
			for(size_t i=0; i<this->myPath.size(); i++){
				if(this->myPath[i] >= 0 && this->myPath[i] < miniMaster.graf.size()){
					Point p;
					p.x = miniMaster.graf[this->myPath[i]][1];
					p.y = miniMaster.graf[this->myPath[i]][0];
					circle(temp ,p ,3,Scalar(175),-1);
				}
			}
			cerr << "3" << endl;
			Size f0;
			resize(temp, temp, f0, 5, 5, CV_INTER_AREA);
			cerr << "4" << endl;
			imshow("coord", temp);
			waitKey(1);
			cerr << "5" << endl;
			if(this->nPulls >= 0 && this->nPulls < 1000){
				cerr << "6" <<endl;
				this->nPulls++;
				if(this->nPulls > 1){ // am I getting tried for the 1st time?
					cerr << "in ucb" << endl;
					int s = this->UCBChildSelect();
					cerr << "out of ucb" << endl;
					//int s = this->eGreedyChildSelect(0.8);
					this->children[s].searchTree(miniMaster);
					cerr << "c.out of search tree" << endl;
					//this->updateMyValue(passedValue);
				}
				else{ // my 1st time!
					cerr << "7" <<endl;
					this->getChildren(miniMaster);
					int s = this->greedyChildSelect(); // choose the best child to represent me for now
					// passedValue = this->children[s].value; // set as passed value
				}
			}
			else{
				cerr << "clearing nPulls" << endl;
				this->nPulls = 0;
			}
		}
		else{
			cerr << "clearing path" << endl;
			this->myPath.clear();
		}
	}
}

void treeNode::evaluateNode(miniGraph &miniMaster){
	Mat tempVisibleMat = Mat::zeros(miniMaster.miniImage.rows, miniMaster.miniImage.cols,CV_8UC1);
	for(size_t i=0; i<this->myPath.size(); i++){
		vector<int> t;
		t.push_back(miniMaster.graf[this->myPath[i]][0]);
		t.push_back(miniMaster.graf[this->myPath[i]][1]);
		miniMaster.observe(t,tempVisibleMat);
	}



	//bitwise_or(this->myVisibleMap, tempVisibleMat, this->myVisibleMap);
	Mat tempRewardMat = Mat::zeros(miniMaster.miniImage.rows,miniMaster.miniImage.cols,CV_8UC1);
	bitwise_and(tempVisibleMat, miniMaster.unknownMat, tempRewardMat);
	this->value = miniMaster.matReward(tempRewardMat) - 50*this->myDepth;

	// for evaluating finishing conditions
	Mat finFlag  = Mat::zeros(miniMaster.miniImage.rows,miniMaster.miniImage.cols,CV_8UC1);
	bitwise_not(tempVisibleMat,tempVisibleMat);
	bitwise_and(miniMaster.unknownMat,tempVisibleMat, finFlag);

	Scalar meanVal = mean(finFlag);


	if(meanVal[0] < 25){
		this->searchComplete = true;
		float pathLength = 0;
		for(size_t i=1; i<this->myPath.size(); i++){
			pathLength += miniMaster.distGraph[this->myPath[i]][this->myPath[i-1]];
		}
		cerr << "pathLength: " << pathLength << endl;
		waitKey(0);
	}

	cerr << "myState / value / depth: " << this->myState << " / " << this->value << " / " << this->myDepth << endl;
	cerr << "myPath: ";
	for(size_t i=0; i<myPath.size(); i++){
		cerr << this->myPath[i ] << " , ";
	}
	cerr << endl;
	imshow("tree", tempRewardMat);
	waitKey(1);
}

void treeNode::getChildren(miniGraph &miniMaster){
	// for all available actions at state
	for(int i=0; i<miniMaster.distGraph.size(); i++){
		if(miniMaster.distGraph[this->myState][i] != INFINITY && i != this->myState){
			cerr << "making child: " << i << endl;
			treeNode a(i, miniMaster, this->myPath, this->myDepth);
			cerr << "made child" << endl;
			this->children.push_back(a);
		}
	}
}

void treeNode::exploitTree(vector<int>& myPath){
	if(this->myDepth == 0){
		myPath.clear();
	}
	myPath.push_back(this->myState);
	cerr << "c: " << this->children.size() << endl;
	if(this->children.size() > 0 && this->children.size() < 10){
		cerr << "into greedy" << endl;
		int s = this->greedyChildSelect();
		cerr << "out of greedy" << endl;
		this->children[s].exploitTree(myPath);
	}
}

void treeNode::deleteTree(){
	if(this->children.size() > 0 && this->children.size() < 10){
		for(size_t i=0; i<this->children.size(); i++){
			this->children[i].deleteTree();
		}
	}
	else{
		this->children.clear();
	}
}

void treeNode::updateMyValue(float passedValue){
	// greedy
	if(this->value < passedValue){
		this->value = passedValue;
	}
	// average
	//this->value = (this->value * this->nPulls + passedValue) / (this->nPulls + 1);
}


int treeNode::UCBChildSelect(){
	cerr << "in in usb" << endl;
	int s = -1; // argmax_(each child) [value(each child) + sqrt(2*ln n(all children) / n(each child]
	float v = -1;
	if(this->children.size() > 0 && this->children.size() < 10){
		for(size_t i=0; i<this->children.size(); i++){
			float cv = children[i].value;
			float nPV = sqrt( log(float(this->nPulls)) / float(children[i].nPulls));
			float tv = cv + nPV;
			if(tv > v){
				v = tv;
				s = i;
			}
		}
	}
	cerr << "ucb nPulls: " << this->nPulls << endl;
	return s;
}

int treeNode::greedyChildSelect(){
	int s = -1;
	float v = -1;
	cerr << "greedy child.size(): " << this->children.size() << endl;
	for(size_t i=0; i<this->children.size(); i++){
		float tv = float(children[i].value);
		if(tv > v){
			v = tv;
			s = i;
		}
	}
	return s;
}

int treeNode::eGreedyChildSelect(float epsilon = 0.5){
	int s = -1;
	if( (rand() % 10000)/10000 > epsilon){
		float v = -1;
		for(size_t i=0; i<this->children.size(); i++){
			float tv = children[i].value;
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

int treeNode::randChildSelect(){
	int s = (rand() % children.size()) + 1;
	return s;
}


int treeNode::simAnnealingChildSelect(float& temp, float cooling = 0.99){
	if(this->myDepth == 0){
		temp = temp * cooling;
	}
	int s = -1;
	float v = -1;
	vector<float> vc;
	for(size_t i=0; i<this->children.size(); i++){
		vc.push_back(children[i].value);
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


treeNode::~treeNode() {
	// TODO Auto-generated destructor stub
}

