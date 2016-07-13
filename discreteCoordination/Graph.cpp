/*
 * Graph.cpp
 *
 *  Created on: Mar 2, 2016
 *      Author: andy
 */


#include "Graph.h"

Point findIntersection(Vec4i w1, Vec4i w2);
float distToLine(Vec4i w, Point a);
Point extendLine(Point a, Point m);
float distToLineSegment(Point p, Point v, Point w);


using namespace cv;
using namespace std;

Graph::Graph(){
	int obsRadius = 40;
	Mat temp =Mat::zeros(2*(obsRadius + 1), 2*(obsRadius + 1), CV_8UC1);
	Point cent;
	cent.x = obsRadius;
	cent.y = obsRadius;
	circle(temp,cent,obsRadius, Scalar(255));

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

void Graph::createGraph(World &gMap, float obsThresh, float comThresh){
	costmap.nRows = gMap.nRows;
	costmap.nCols = gMap.nCols;
	this->initializeCostMap();

	this->frntsExist = true;
	this->image = gMap.image;
	this->tempImage = gMap.image;
	this->obsThresh = obsThresh;
	this->comThresh = comThresh;
}

Mat Graph::createMiniMapImg(){
	Mat temp = Mat::zeros(costmap.nCols, costmap.nRows,CV_8UC1);
	for(int i=0; i<costmap.nRows; i++){
		for(int j=0; j<costmap.nCols; j++){
			temp.at<uchar>(i,j,0) = costmap.cells[i][j];
		}
	}
	return temp;
}

void Graph::findFrontiers(){
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

vector<Point> Graph::getImagePoints(Mat &image){
	vector<Point> temp;
	for(int i=0; i<image.rows; i++){
		for(int j=0; j<image.cols; j++){
			Scalar intensity = image.at<uchar>(i,j);
			if(intensity[0] > 100){
				Point t;
				t.x = j;
				t.y = i;
				temp.push_back(t);
			}
		}
	}
	return temp;
}



Mat Graph::getObstaclesImage(Costmap &costmap){
	Mat temp = Mat::zeros(costmap.nRows,costmap.nCols,CV_8UC1);
	for(int i=0; i<costmap.nRows; i++){
		for(int j=0; j<costmap.nCols; j++){
			if(costmap.cells[i][j] > 100){ // free space with inflation
				temp.at<uchar>(i,j,0) = 255;
			}
			else{
				temp.at<uchar>(i,j,0) = 0;
			}
		}
	}
	//imshow("obstacles map", temp);
	return temp;
}

Mat Graph::getFreeSpaceImage(Costmap &costmap){
	Mat temp = Mat::zeros(costmap.nRows, costmap.nCols,CV_8UC1);
	for(int i=0; i<costmap.nRows; i++){
		for(int j=0; j<costmap.nCols; j++){
			if(costmap.cells[i][j] <50){ // free space with inflation
				temp.at<uchar>(i,j,0) = 255;
			}
			else{
				temp.at<uchar>(i,j,0) = 0;
			}
		}
	}
	//imshow("freespace map", temp);
	return temp;
}

Mat Graph::getFrontiersImage(Costmap & costmap){
	Mat temp = Mat::zeros(costmap.nRows, costmap.nCols,CV_8UC1);
	for(int i=0; i<costmap.nRows; i++){
		for(int j=0; j<costmap.nCols; j++){
			if(costmap.cells[i][j] == 50){ // free space with inflation
				temp.at<uchar>(i,j,0) = 255;
			}
			else{
				temp.at<uchar>(i,j,0) = 0;
			}
		}
	}
	//imshow("Frontiers map", temp);
	return temp;
}

int Graph::getMaxIndex(vector<float> value){
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

int Graph::getMinIndex(vector<float> value){
	int mindex;
	float minValue = INFINITY;
	for(int i=0; i<(int)value.size(); i++){
		if(value[i] < minValue){
			mindex = i;
			minValue  = value[i];
		}
	}
	return mindex;
}

void Graph::showCostMapPlot(int index){
	namedWindow("graph::costMap",WINDOW_NORMAL);
	imshow("graph::costMap", this->tempImage);
	waitKey(1);
}

void Graph::addAgentToPlot(Scalar color, vector<vector<int> > myPath, vector<int> cLoc){
	circle(this->tempImage,Point(cLoc[1],cLoc[0]),2,color,-1);
	for(size_t i=1; i<myPath.size(); i++){
		Point a = Point(myPath[i][1],myPath[i][0]);
		Point b = Point(myPath[i-1][1],myPath[i-1][0]);
		line(this->tempImage,a,b,color,1);
	}
}

void Graph::buildCostMapPlot(Costmap &costmap){
	Vec3b color;
	this->tempImage = Mat::zeros(costmap.nRows, costmap.nCols,CV_8UC3);
	for(int i=0; i<costmap.nRows; i++){
		for(int j=0; j<costmap.nCols; j++){
			if(costmap.cells[i][j] < 10){ // free space with inflation
				color[0] = 255;
				color[1] = 255;
				color[2] = 255;
				this->tempImage.at<Vec3b>(i,j) = color;
			}
			else if(costmap.cells[i][j] > 200){ // obstacle
				color[0] = 0;
				color[1] = 0;
				color[2] = 0;
				this->tempImage.at<Vec3b>(i,j) = color;
			}
			else if(costmap.cells[i][j] > 100){ // unknown space
				color[0] = 127;
				color[1] = 127;
				color[2] = 127;
				this->tempImage.at<Vec3b>(i,j) = color;
			}
		}
	}
	color[0] = 0;
	color[1] = 0;
	color[2] = 127;
	for(size_t i=0; i<(int)this->frontiers.size(); i++){
		for(size_t j=0; j<this->frontiers[i].members.size(); j++){
			this->tempImage.at<Vec3b>(this->frontiers[i].members[j][0],this->frontiers[i].members[j][1]) = color;
		}
	}
	color[0] = 0;
	color[1] = 0;
	color[2] = 255;
	for(size_t i=0; i<this->frontiers.size(); i++){
		this->tempImage.at<Vec3b>(this->frontiers[i].centroid[0],this->frontiers[i].centroid[1]) = color;
	}
}


float distToLineSegment(Point p, Point v, Point w){
	float l = pow(v.x - w.x,2) + pow(v.y-w.y,2);
	if(l==0){ return sqrt(pow(v.x - p.x,2) + pow(v.y-p.y,2) ); } // v==w
	float t = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l;
	if(t > 1){
		t = 1;
	}
	else if(t < 0){
		t = 0;
	}
	int xl = v.x + t * (w.x - v.x);
	int yl = v.y + t * (w.y - v.y);
	return sqrt( pow(p.x - xl,2) + pow(p.y-yl,2) );
}

Point extendLine(Point a, Point m){
	float dx = m.x - a.x;
	float dy = m.y - a.y;
	Point p;
	p.x = m.x + dx;
	p.y = m.y + dy;
	return(p);
}

float distToLine(Vec4i w, Point a){
	float x1 = w[0];
	float y1 = w[1];
	float x2 = w[2];
	float y2 = w[3];

	float x0 = a.x;
	float y0 = a.y;

	float denom = sqrt(pow(x2-x1,2) + pow(y2-y1,2));
	if(denom != 0){
		float dist = abs((x2-x1)*(y1-y0)-(x1-x0)*(y2-y1))/denom;
		return dist;
	}
	else{
		return(-1);
	}
}

Point findIntersection(Vec4i w1, Vec4i w2){
	float x1 = w1[0];
	float y1 = w1[1];
	float x2 = w1[2];
	float y2 = w1[3];

	float x3 = w2[0];
	float y3 = w2[1];
	float x4 = w2[2];
	float y4 = w2[3];

	float denom = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4);

	if(denom != 0){
		Point p;
	    p.x = ((x1*y2-y1*x2)*(x3-x4) - (x3*y4-y3*x4)*(x1-x2))/denom;
	    p.y = ((x1*y2-y1*x2)*(y3-y4) - (x3*y4-y3*x4)*(y1-y2))/denom;
	    return(p);
	}
	else{
		Point p;
		p.x = -1;
		p.y = -1;
		return(p);
	}
}

 int Graph::matReward(Mat &in){
 	// get Mat entropy
 	int observed = 0;
 	for(int i=0; i<in.cols; i++){
 		for(int j=0; j<in.rows; j++){
 			if(in.at<uchar>(i,j,0) > 0){
 				observed++;
 			}
 		}
 	}
 	return observed;
 }


 void Graph::simulateObservation(vector<int> pose, Mat &viewMat, Mat &costMat){
 	// make perimeter of viewing circle fit on image
 	for(size_t i=0; i<this->viewPerim.size(); i++){
 		int px = pose[0] + this->viewPerim[i][0];
 		int py = pose[1] + this->viewPerim[i][1];

 		bool flag = true;
 		while(flag){
 			flag = false;
 			if(px < 0){
 				float dx = pose[0] - px;
 				float dy = pose[1] - py;

 				float m = dy / dx;
 				float b = pose[1]-m*pose[0];
 				px = 0;
 				py = b;
 				flag = true;
 			}
 			else if(px >= viewMat.cols){
 				float dx = pose[0] - px;
 				float dy = pose[1] - py;

 				float m = dy / dx;
 				float b = pose[1]-m*pose[0];
 				px = viewMat.cols-1;
 				py = m*px + b;
 				flag = true;
 			}
 			else if(py < 0){ // NOTE should this be just if?
 				float dx = pose[0] - px;
 				float dy = pose[1] - py;

 				float m = dy / dx;
 				float b = pose[1]-m*pose[0];
 				py = 0;
 				px = (py-b)/m;
 				flag = true;
 			}
 			else if(py >= viewMat.rows){
 				float dx = pose[0] - px;
 				float dy = pose[1] - py;

 				float m = dy / dx;
 				float b = pose[1]-m*pose[0];
 				py = viewMat.rows-1;
 				px = (py-b)/m;
 				flag = true;
 			}
 		}

 		// check visibility to all points on circle
 		float dx = px - pose[0];
 		float dy = py - pose[1];
 		if(dx != 0){
 			if(dx > 0){
 				float m = dy/dx;
 				float b = pose[1]-m*pose[0];

 				int y0 = pose[1];
 				for(int x0 = pose[0]; x0 < px; x0++){
 					y0 = m*x0+b;
 					if(costMat.at<uchar>(x0,y0) > 10){
 						break;
 					}
 					else if(costMat.at<uchar>(x0,y0) == 2){
 						viewMat.at<uchar>(x0,y0,0) = 255;
 					}
 				}
 			}
 			else{
 				float m = dy/dx;
 				float b = pose[1]-m*pose[0];

 				int y0 = pose[1];
 				for(int x0 = pose[0]; x0 > px; x0--){
 					y0 = m*x0+b;
 					if(costMat.at<uchar>(x0,y0) > 10){
 						break;
 					}
 					else if(costMat.at<uchar>(x0,y0) == 2){
 						viewMat.at<uchar>(x0,y0,0) = 255;
 					}
 				}
 			}
 		}
 		else{
 			if(dy > 0){
 				int x0 = pose[0];
 				for(int y0 = pose[1]; y0 < py; y0++){
 					if(costMat.at<uchar>(x0,y0) > 10){
 						break;
 					}
 					else if(costMat.at<uchar>(x0,y0) == 2){
 						viewMat.at<uchar>(x0,y0,0) = 255;
 					}
 				}
 			}
 			else{
 				int x0 = pose[0];
 				for(int y0 = pose[1]; y0 > py; y0--){
 					if(costMat.at<uchar>(x0,y0) > 10){
 						break;
 					}
 					else if(costMat.at<uchar>(x0,y0) == 2){
 							viewMat.at<uchar>(x0,y0,0) = 255;
 					}
 				}
 			}
 		}
 	}
 }

 void Graph::findPoseSet(Mat &costMat){

 	cout << "Graph::getposeSet::poseSet.size(): " << poseSet.size() << endl;

 	Mat cView = Mat::zeros(costMat.size(), CV_8UC1);
 	int cReward = 0;

 	int gReward = 0;
 	for(int i=0; i<costMat.cols; i++){
 		for(int j=0; j<costMat.rows; j++){
 			if(costMat.at<uchar>(i,j) == 2){
 				gReward++;
 			}
 		}
 	}
 	while(true){
 		if(this->poseSet.size() == 0){ // initialize this->poseSet
 			while(this->poseSet.size() == 0){
 				vector<int> ps;
 				ps.push_back( rand() % costMat.cols );
 				ps.push_back( rand() % costMat.rows );
 				if(costMat.at<uchar>(ps[0],ps[1]) < 10){
 					Mat tView = Mat::zeros(cView.size(), CV_8UC1);
 					this->simulateObservation(ps, tView, costMat);
 					int tReward = this->matReward(tView);
 					if(tReward > 0){
 						this->poseSet.push_back(ps);
 					}
 				}
 			}
 		}
 		else{
 			vector<vector<int> > tSet = poseSet;
 			this->poseSet.clear();
 			for(size_t i=0; i<tSet.size(); i++){
 				if(costMat.at<uchar>(tSet[i][0],tSet[i][1]) < 10){
 					Mat tView = Mat::zeros(cView.size(), CV_8UC1);
 					this->simulateObservation(tSet[i], tView, costMat);
 					int tReward = this->matReward(tView);
 					if(tReward > 0){
 						this->poseSet.push_back(tSet[i]);
 					}
 				}
 			}
 		}
 		if(poseSet.size() > 0){
 			break;
 		}
 	}

 	// create poseSetView
 	for(size_t i=0; i<this->poseSet.size(); i++){
 		Mat tView = Mat::zeros(cView.size(), CV_8UC1);
 		this->simulateObservation(this->poseSet[i], tView, costMat);
 		bitwise_or(cView, tView, cView);
 	}
 	cReward = this->matReward(cView);

 	float entropy = float(cReward) / float(gReward);

 	// erase poses that don't contribute
 	int pi = rand() % poseSet.size();
 	vector<vector<int> > tSet = poseSet;
 	tSet.erase(tSet.begin()+pi);
 	// create cView minus selected node
 	Mat pView =  Mat::zeros(cView.size(), CV_8UC1);
 	for(size_t i=0; i<tSet.size(); i++){
 		Mat tView = Mat::zeros(cView.size(), CV_8UC1);
 		this->simulateObservation(tSet[i], tView, costMat);
 		bitwise_or(pView, tView, pView);
 	}
 	int pReward = this->matReward(pView);
 	float pEntropy = float(pReward) / float(gReward);
 	if(pEntropy >= 0.95*entropy){
 		this->poseSet = tSet;
 	}

 	int cnt = 0;
 	while(entropy < 1 && cnt < 1000){
 		cnt++;
 		// add new pose
 		if(entropy < (rand() % 1000) / 500){
 			//cout << "graph::poseSet::addPose" << endl;
 			while(true){
 				vector<int> ps;
 				ps.push_back( rand() % costMat.cols );
 				ps.push_back( rand() % costMat.rows );
 				if(costMat.at<uchar>(ps[0],ps[1]) < 10){ // free or inferred free
 					Mat tView = Mat::zeros(cView.size(), CV_8UC1);
 					this->simulateObservation(ps, tView, costMat);
 					bitwise_or(cView, tView, tView);
 					int tReward = this->matReward(tView);
 					if(tReward > cReward){
 						this->poseSet.push_back(ps);
 						break;
 					}
 				}
 			}
 		}
 		else{
 		// erase pose
 			if(rand() % 1000 > 500){
 				//cout << "graph::poseSet::gradientPose" << endl;
 				int pi = rand() % poseSet.size();
 				vector<vector<int> > tSet = poseSet;
 				tSet.erase(tSet.begin()+pi);
 				// create cView minus selected node
 				Mat pView =  Mat::zeros(cView.size(), CV_8UC1);
 				for(size_t i=0; i<tSet.size(); i++){
 					Mat tView = Mat::zeros(cView.size(), CV_8UC1);
 					this->simulateObservation(tSet[i], tView, costMat);
 					bitwise_or(pView, tView, pView);
 				}
 				int pReward = this->matReward(pView);

 				// get +/- x/y nbrs
 				vector<vector<int> > tP;
 				tP.push_back(poseSet[pi]);
 				vector<int> ps = poseSet[pi];
 				ps[1]++;
 				tP.push_back(ps);
 				ps = poseSet[pi];
 				ps[1]--;
 				tP.push_back(ps);
 				ps = poseSet[pi];
 				ps[0]++;
 				tP.push_back(ps);
 				ps = poseSet[pi];
 				ps[0]--;
 				tP.push_back(ps);
 				int maxi;
 				int maxv = -1;
 				for(size_t i=0; i<tP.size(); i++){
 					Mat tView = Mat::zeros(cView.size(), CV_8UC1);
 					this->simulateObservation(tP[i], tView, costMat);
 					bitwise_or(pView, tView, tView);
 					int tReward = this->matReward(tView);
 					if(tReward > maxv){
 						maxv = tReward;
 						maxi = i;
 					}
 				}
 				tSet.push_back(tP[maxi]);
 				this->poseSet = tSet;
 			}
 			else{ // move pose with +/- x/y gradient
 				//cout << "graph::poseSet::erasePose" << endl;
 				int pi = rand() % poseSet.size();
 				vector<vector<int> > tSet = poseSet;
 				tSet.erase(tSet.begin()+pi);

 				// create cView minus selected node
 				Mat pView =  Mat::zeros(cView.size(), CV_8UC1);
 				for(size_t i=0; i<tSet.size(); i++){
 					Mat tView = Mat::zeros(cView.size(), CV_8UC1);
 					this->simulateObservation(tSet[i], tView, costMat);
 					bitwise_or(pView, tView, pView);
 				}
 				int pReward = this->matReward(pView);
 				float pEntropy = float(pReward) / float(gReward);
 				if(pEntropy >= 0.95*entropy){
 					this->poseSet = tSet;
 				}
 			}
 		}
 		// get cReward
 		for(size_t i=0; i<this->poseSet.size(); i++){
 			Mat tView = Mat::zeros(cView.size(), CV_8UC1);
 			this->simulateObservation(this->poseSet[i], tView, costMat);
 			bitwise_or(cView, tView, cView);
 		}
 		cReward = this->matReward(cView);
 		entropy = float(cReward) / float(gReward);
 	}

 	cout << "Graph::getposeSet::entropy/poseSet.size()/cnt: " << entropy << " / " << poseSet.size() << " / " << cnt << endl;
 	for(size_t i=0; i<this->poseSet.size(); i++){
 		cout << this->poseSet[i][0] << " , " << this->poseSet[i][1] << endl;
 		circle(cView,Point(poseSet[i][1],poseSet[i][0]),2,Scalar(127),-1);
 	}
 	namedWindow("Graph::getposeSet::cView", WINDOW_NORMAL);
 	imshow("Graph::getposeSet::cView", cView);

 	waitKey(0);

 }

Graph::~Graph() {

}




