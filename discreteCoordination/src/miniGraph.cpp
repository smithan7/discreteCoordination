/*
 * miniGraph.cpp
 *
 *  Created on: Mar 14, 2016
 *      Author: andy
 */

#include "miniGraph.h"

using namespace cv;
using namespace std;

miniGraph::miniGraph(){
	// TODO Auto-generated constructor stub
	int obsRadius = 80;
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

/*
void miniGraph::buildTree(){

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
	cerr << "myState: " << myTree.myState << endl;
	cerr << "children.size(): " << myTree.children.size() << endl;
	for(size_t i=0; i < myTree.children.size(); i++){
		cerr << "   " << myTree.children[i].myState << endl;
	}
	waitKey(0);
}
*/

float miniGraph::matReward(Mat &in){
	// get Mat entropy
	float observed = 0;
	for(size_t i=0; i<in.cols; i++){
		for(size_t j=0; j<in.rows; j++){
			if(in.at<uchar>(i,j,0) > 0){
				observed++;
			}
		}
	}
	return observed;
}

Mat miniGraph::breadthFirstSearchFindRoom(Mat &src, vector<int> pt){

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
				cerr << endl;
			}
		}
	}

	// add initial point to the closedSet
	cSet.push_back(cPt);

	// while there are still points in the openSet
	while(oSet.size() > 0){
		// add current point to closedSet and remove from openSet
		cPt = oSet[oSet.size()-1];
		cerr << cPt[0] << " < " << cPt[1] << endl;
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


void miniGraph::growObstacles(){
	// identify obstacles of length > x
	// determine obstacle orientation and curvature
	// extend obstacle per both
}

bool miniGraph::lineTraversabilityCheck(Mat &tSpace, vector<int> sPt, vector<int> fPt, int fValue){

	if(abs(fPt[0] - sPt[0]) == abs(fPt[1] - sPt[1])){ // larger change in x direction, count along x
		if(fPt[0] < sPt[0]){ // set order right
			vector<int> t = sPt;
			sPt = fPt;
			fPt = t;
			cerr << "inv" << endl;
		}

		float m = float(fPt[1] - sPt[1]) / float(fPt[0] - sPt[0]);
		float b = float(fPt[1]) - m*float(fPt[0]);

		cerr << "fPt: " << fPt[1] << " , " << fPt[0] << endl;
		cerr << "sPt: " << sPt[1] << " , " << sPt[0] << endl;
		cerr << "x: " << m << " , " << b << endl;

		Mat temp = tSpace;

		for(int x = sPt[0]+1; x<fPt[0]-1; x++){
			float tx = x;
			float ty = m*tx+b;
			int y = ty;
			temp.at<uchar>(y,x,0) = 127;
			imshow("zzz", temp);
			//waitKey(0);

			if(tSpace.at<uchar>(y,x,0) != fValue){
				cerr << "false" << endl;
				return false;
			}
		}
		cerr << "return true" << endl;
		return true;
	}
	else{
		if(fPt[1] < sPt[1]){ // set order right
			vector<int> t = sPt;
			sPt = fPt;
			fPt = t;
			cerr << "inv" << endl;
		}
		float m = float(fPt[0] - sPt[0]) / float(fPt[1] - sPt[1]);
		float b = float(fPt[0]) - m*float(fPt[1]);

		cerr << "y: " << m << " , " << b << endl;

		Mat temp = tSpace;

		for(int x = sPt[1]+1; x<fPt[1]-1; x++){
			int y = round(m*x+b);
			temp.at<uchar>(y,x,0) = 127;
			imshow("zzz", temp);
			//waitKey(0);

			if(tSpace.at<uchar>(x,y,0) != fValue){
				cerr << "false" << endl;
				return false;
			}
		}
		cerr << "return true" << endl;
		return true;
	}
}

void miniGraph::extractInferenceContour(){
	Mat temp;
	bitwise_or(this->obstacleMat, this->inferenceMat, temp);
	bitwise_or(temp, this->freeMat, temp);
	bitwise_not(temp,temp);

	imshow("q", temp);
}


void miniGraph::growFrontiers(vector<frontier> frnt){

	// find all frontiers with traversable bath through free and observed space between them using line check

	// find all members of each frontier and frontier orientation^-1 and add all to oSet
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

	// find all frontiers with traversable bath through free and observed space between them using line check
	vector<vector<frontier> > mFrnts;
	for(size_t i=0; i<frnt.size()-1; i++){
		vector<frontier> t;
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
	// all points in frontier extend in direction of orient one unit as long as it is free space
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

void miniGraph::invertImageAroundPt(Mat &src, Mat &dst, vector<int> cLoc){

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
	waitKey(0);
}


void miniGraph::getNodeValues(){
	this->nodeValue.erase(this->nodeValue.begin(), this->nodeValue.end());
	for(int i=0; i<this->nmNodes; i++){
		this->nodeValue.push_back(this->nodeReward[i] - this->nodeCost[i]);
	}
}

int miniGraph::getMaxIndex(vector<float> value){
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

void miniGraph::getNodeCosts(int cNode){
	this->nodeCost.erase(this->nodeCost.begin(), this->nodeCost.end());
	for(int i=0; i<(int)this->nmNodes; i++){// for each node
		this->nodeCost.push_back(this->distGraph[cNode][i]);//.push_back(this->aStarDist(cNode,i)); // get A* cost to each frontier
	}
	cerr << "nodeCosts: ";
	for(int i=0;i<this->nmNodes; i++){
		cerr << "nodeCost[" << i << "]: " << this->nodeCost[i] << endl;
	}
}

void miniGraph::getNodeRewards(){
	this->nodeReward.erase(this->nodeReward.begin(), this->nodeReward.end());
	this->nodeFrontiers.erase(this->nodeFrontiers.begin(), this->nodeFrontiers.end());
	for(int i=0; i<this->nmNodes; i++){// for each frontier
		this->nodeReward.push_back(0);
		vector<int> t;
		this->nodeFrontiers.push_back(t);
	}
	for(int i=0; i<(int)this->frontiers.size(); i++){// for each frontier
		vector<int> t;
		t.push_back(frontiers[i][1]);
		t.push_back(frontiers[i][0]);
		int a = findNearestNode(t); // find node closest to frontier
		this->nodeFrontiers[a].push_back(i);
		this->nodeReward[a] += 50; // sub in froniter value
	}

	for(int i=0;i<this->nmNodes; i++){
		cerr << "nodeFrontiers[" << i << "]: ";
		for(int j=0; j<this->nodeFrontiers[i].size(); j++){
			cerr << this->nodeFrontiers[i][j] << ", ";
		}
		cerr << endl;
	}
}

void miniGraph::importFrontiers(vector<vector<int> > frontierList){ // bring in frontiers to miniGraph
	this->frontiers.clear();
	this->frontiers = frontierList;
}

void miniGraph::importUAVLocations(vector<vector<int> > cLocList){ // bring in UAV locations to miniGraph
	this->cLocListMap.clear();
	this->cLocListMap = cLocList;
}

int miniGraph::findNearestNode(vector<int> in){
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

float miniGraph::aStarDist(int strt, int gl){
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
	fScore[strt] = gScore[strt] + this->distGraph[strt][gl]; // calc score of open set
	int foo = 1;
	int finishFlag = 0;
	while(foo>0 && finishFlag == 0){
		/////////////////// this finds node with lowest fScore and makes current
		float min = INFINITY;
		int iMin = 0;
		for(int i=0;i<this->nmNodes;i++){
			//cout << i << "/" << oSet[i] << "/" << fScore[i] << "/" << min << endl;
			if(oSet[i] > 0 && fScore[i] < min){
				min = fScore[i];
				iMin = i;
			}
		}
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
			if(this->distGraph[current][nbr] > 0){ // find all adj neighbors that are observed
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
				fScore[nbr] = gScore[nbr] + this->distGraph[nbr][gl];
			}
		}
		/////////////// end condition for while loop, check if oSet is empty
		foo = 0;
		for(int i=0;i<this->nmNodes;i++){
			foo+= oSet[i];
		}
		//foo--;
	}
	return INFINITY;
}

void miniGraph::observe(vector<int> cLoc, Mat &viewMat){
	// make perimeter of viewing circle fit on image
	for(size_t i=0; i <this->viewPerim.size(); i++){
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
	}
	//imshow("viewMap", viewMat);
	//waitKey(1);
}

void miniGraph::getInferenceMat(Mat &inputMat){
	this->inferenceMat = inputMat;
}

void miniGraph::getUnobservedMat(Mat &inputMat){
	Scalar white = 255;
		Scalar gray = 127;

		// 0 = free space
		// 50 = frontier
		// 100 = unknown
		// Infinity = obstacle

		this->unknownMat = Mat::zeros(inputMat.rows, inputMat.cols,CV_8UC1);

		Mat mapImage = Mat::zeros(inputMat.rows, inputMat.cols,CV_8UC3);
		Vec3b white3; white3[0] = 255; white3[1] = 255; white3[2] = 255;
		Vec3b red; red[0] = 0; red[1] = 0; red[2] = 255;
		Vec3b blue; blue[0] = 255; blue[1] = 0; blue[2] = 0;
		Vec3b gray3; gray3[0] = 127; gray3[1] = 127; gray3[2] = 127;

		for(size_t i = 0; i<inputMat.cols; i++){
			for(size_t j =0; j<inputMat.rows; j++){
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


void miniGraph::createMiniGraph(Mat &inputMat, int minX, int minY, int maxX, int maxY){
	Scalar white = 255;
	Scalar gray = 127;

	// 0 = free space
	// 50 = frontier
	// 100 = unknown
	// Infinity = obstacle

	this->obstacleMat = Mat::zeros(inputMat.rows, inputMat.cols,CV_8UC1);
	this->freeMat = Mat::zeros(inputMat.rows, inputMat.cols,CV_8UC1);
	this->frontierMat = Mat::zeros(inputMat.rows, inputMat.cols,CV_8UC1);
	this->unknownMat = Mat::zeros(inputMat.rows, inputMat.cols,CV_8UC1);

	Mat mapImage = Mat::zeros(inputMat.rows, inputMat.cols,CV_8UC3);
	Vec3b white3; white3[0] = 255; white3[1] = 255; white3[2] = 255;
	Vec3b red; red[0] = 0; red[1] = 0; red[2] = 255;
	Vec3b blue; blue[0] = 255; blue[1] = 0; blue[2] = 0;
	Vec3b gray3; gray3[0] = 127; gray3[1] = 127; gray3[2] = 127;

	// 0 = free
	// 1 = inferred free
	// 255 = wall
	// 254 = inferred wall

	for(size_t i = 0; i<inputMat.cols; i++){
		for(size_t j =0; j<inputMat.rows; j++){
			Scalar intensity =  inputMat.at<uchar>(i,j,0);
			if(intensity[0] <= 1){ // free space or inferred free space
				this->freeMat.at<uchar>(i,j,0) = 255;
				mapImage.at<Vec3b>(i,j) = white3;
			}
			else if(intensity[0] == 50){ // frontier
				this->frontierMat.at<uchar>(i,j,0) = 255;
				mapImage.at<Vec3b>(i,j) = blue;
			}
			else if(intensity[0] == 100){ //
				this->unknownMat.at<uchar>(i,j,0) = 255;
				mapImage.at<Vec3b>(i,j) = gray3;
			}
			else if(intensity[0] > 100){
				this->obstacleMat.at<uchar>(i,j,0) = 100;
				mapImage.at<Vec3b>(i,j) = white3;
			}
		}
	}

	threshold(this->freeMat,this->miniImage,10,255,CV_THRESH_BINARY);
	imshow("q-miniImage", this->miniImage);


	// need to apply a mask to this so it only updates areas in view of UAVs
	this->thinning(this->miniImage,this->miniImage, minX, minY, maxX, maxY);

	vector<vector<int> > miniGraphNodes;
	for(int i=0; i<this->miniImage.cols; i++){
		for(int j=0; j<this->miniImage.rows; j++){
			Scalar intensity =  this->miniImage.at<uchar>(i,j);
			if(intensity[0] == 255){
				vector<int> t;
				t.push_back(i);
				t.push_back(j);
				//this->graf.push_back(t);
				miniGraphNodes.push_back(t);
			}
		}
	}

	this->graf.clear();
	this->grafSpacing = 5;
	// city block distance for graph
	this->findCityBlockDistanceNodes(miniGraphNodes); // add them to graf ;
	this->nmNodes = this->graf.size();
	this->cityBlockDistanceNodeConnections(); // find connections in graf;

	// this->findPointOfInterestNodes(); // add add them to graf
	// this->breadthFirstSearchAssembleMiniGraph(); // find connections

    for(size_t i=0; i<this->graf.size(); i++){
    	Point temp;
        temp.x = this->graf[i][1];
        temp.y = this->graf[i][0];
        circle(this->miniImage,temp,1,gray,-1,8);
    }
    for(size_t i=0; i<this->distGraph.size(); i++){
    	for(size_t j=0; j<this->distGraph[i].size(); j++){
    		if(this->distGraph[i][j] < 10 * this->grafSpacing){
    			Point a,b;
    			a  = Point(this->graf[i][1], this->graf[i][0]);
    			b = Point(this->graf[j][1], this->graf[j][0]);
    			line(this->miniImage,a,b,white,1);
    		}
    		//cerr << this->distGraph[i][j] << ", ";
    	}
    	//cerr << endl;
    }
    //Size f0;
    //Mat tempView;
    //resize(this->miniImage, tempView,f0, 5.0, 5.0);
    //imshow("Global miniMap' ", tempView);
	//waitKey(1);
	this->cNode = this->findNearestNode(this->cLocList[0]);

}

void miniGraph::drawCoordMap(vector<int> cLoc){
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
		temp.x = this->graf[i][0]*10;
		temp.y = this->graf[i][1]*10;
		circle(coordGraph,temp,1,white,-1,8);
		char str[50];
		sprintf(str,"%d",i);
		putText(coordGraph, str, temp, FONT_HERSHEY_PLAIN,2,green);
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

	for(int i=0;i<this->frontiers.size(); i++){
		Point temp;
		temp.x = this->frontiers[i][1]*10;
		temp.y = this->frontiers[i][0]*10;
		circle(coordGraph,temp,5,red,-1,8);
	}

	Point temp;
	temp.x = cLoc[1]*10;
	temp.y = cLoc[0]*10;
	circle(coordGraph,temp,10,blue,-1,8);

	imshow("coordGraph", coordGraph);
	waitKey(1);
}

bool miniGraph::bresenhamLineCheck(vector<int> cLoc, vector<int> cPt){
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


void miniGraph::cityBlockDistanceNodeConnections(){
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
			if(this->euclidianDist(graf[i],graf[j]) < 2.5 * this->grafSpacing){
				this->distGraph[i][j] = this->euclidianDist(graf[i],graf[j]);
			}
		}
	}
}


void miniGraph::findCityBlockDistanceNodes(vector<vector<int> > miniGraphNodes){
	this->graf.clear();
	this->graf.push_back(miniGraphNodes[0]);
	for(size_t i=1; i<miniGraphNodes.size(); i++){
		bool tFlag = true;
		for(size_t j=0; j<this->graf.size(); j++){
			if(this->euclidianDist(miniGraphNodes[i],this->graf[j]) < this->grafSpacing){
				tFlag = false;
			}
		}
		if(tFlag){
			this->graf.push_back(miniGraphNodes[i]);

		}
	}
}

void miniGraph::cornerFinder(Mat &inputMat){
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

bool miniGraph::bisectionCheck(vector<int> a, vector<int> b){
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

float miniGraph::cityBlockDist(vector<int> a, vector<int> b){
	float d = abs(a[0] - b[0]) + abs(a[1]+b[1]);
	return d;
}

float miniGraph::euclidianDist(vector<int> a, vector<int> b){
	float d = sqrt(pow(a[0]-b[0],2) + pow(a[1] - b[1],2));
	return d;
}

void miniGraph::thinning(const Mat& src, Mat& dst, int minX, int minY, int maxX, int maxY){
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

void miniGraph::thinningIteration(Mat& img, int iter){
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

miniGraph::~miniGraph() {
	// TODO Auto-generated destructor stub
}

void miniGraph::breadthFirstSearchAssembleMiniGraph(){
	// breadth first search a random node to all other nodes to get travel distances
	// during search check if each location is a node and log distance then remove that branch from the search
	// start search again from next nodes, think of this as exploring all edges attached to one node to get dist;

	// create a list of nodes
	vector<vector<int> > nLoc;
	for(int i=0; i<this->nmNodes; i++){
		vector<int> tempList;
		tempList.push_back(this->graf[i][0]);
		tempList.push_back(this->graf[i][1]);
		//cerr << this->graf[i].x << " & " << this->graf[i].y << endl;
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
	//				cerr << "*";
	//			}
	//			else{
	//				cerr << flag[i][j];
	//			}
	//		}
	//	}
	//	cerr << endl;
	//}


	//cerr << "There are " << this->nmNodes << endl;
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

		//cerr << "BEGIN ITERATION WITH SEED " << seed << " /////// nLoc: " << nLoc[seed][0] << "< " << nLoc[seed][1] << endl;
		while((int)oSet.size() > 0){
			//cerr << "   oSet: ";
			//for(int k=0; k<(int)oSet.size(); k++){
			//	cerr << oSet[k][0] << "," << oSet[k][1] << "; ";
			//}
			//cerr << endl;
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
			//cerr << "minDex: " << minLoc[0] << "," << minLoc[1] << endl;

			// is mindex at an undiscovered node?
			//cerr << "checking if minDex is at an undiscovered Node" << endl;
			for(int j = -2; j<3; j++){
				for(int k=-2; k<3; k++){

					for(int i=0; i<this->nmNodes; i++){ // am I at a new node?
						if(nLoc[i][0] == minLoc[0]+j && nLoc[i][1] == minLoc[1]+k && i != seed){
				//			cerr << "  Found node " << i << " from seed " << seed << " with a dist of " << dist[mindex]+sqrt(pow(j,2)+pow(k,2)) << endl;
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
			//cerr << "checking if minDex is near a discovered node" << endl;
			//for(int i=0; i<(int)foundNode.size(); i++){
			//	if(sqrt(pow(foundNode[i][0]-minLoc[0] && foundNode[i][1] == minLoc[1],2)) < 1){ // am I near a node I have found before?
			//		cerr << "   near a discovered node" << endl;
			//		skip = true;
			//	}
			//}

			// check minDex's nbrs to see if they should be added to the open set
			//cerr << "checking nbrs" << endl;

			//for(int i = -2; i<3; i++){
			//	for(int j=-2; j<3; j++){
			//		if(i == 0 && j == 0){
			//			cerr << "*";
			//		}
			//		else{
			//			cerr << flag[minLoc[0] + i][minLoc[1] + j];
			//		}
			//	}
			//	cerr << endl;
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
							//cerr << "   found a nbr: " << minLoc[0] + i << "," << minLoc[1] + j << " at dist: " <<  dist[mindex] + sqrt(pow(i,2)+pow(j,2)) << endl;
						}
					}
				}
				//cerr << "out" << endl;
			}
			// move the current node out of open set and into closed set
			//cerr << "moving minDex to closed set" << endl;
			vector<int> ml;
			ml.push_back(minLoc[0]);
			ml.push_back(minLoc[1]);
			cSet.push_back(ml);

			//cerr << "   cSet: ";
			//for(int k=0; k<(int)cSet.size(); k++){
			//	cerr << cSet[k][0] << "," << cSet[k][1] << "; ";
			//}
			//cerr << endl;

			//cerr << "   oSet: ";
			//for(int k=0; k<(int)oSet.size(); k++){
			//	cerr << oSet[k][0] << "," << oSet[k][1] << "; ";
			//}
			//cerr << endl;

			oSet.erase(oSet.begin()+mindex,oSet.begin()+mindex+1);
			dist.erase(dist.begin()+mindex,dist.begin()+mindex+1);
		}
	}

	for(int i=0; i<this->nmNodes; i++){
		this->distGraph[i][i] = 0;
	}

	//cerr << "DISTGRAPH:" << endl;
	//for(int i=0; i<this->nmNodes; i++){
	//	for(int j=0; j<this->nmNodes; j++){
	//		cerr << floor(100*this->distGraph[i][j])/100 << " , ";
	//	}
	//	cerr << endl;
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
	waitKey(0);
	*/
}

void miniGraph::findPointOfInterestNodes(){

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

	this->graf.erase(this->graf.begin(), this->graf.end());
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
				//cerr << "vec: ";
				//for(int i=0; i<10; i++){
				//	cerr << outerEdge[i] << ",";
				//}
			   //cerr << endl;
			   //cerr << "edgeDetector: " << edgeDetector << endl;
			   //cerr << (*bb != 0) << "," << (*bc != 0) << "," << (*bd != 0) << endl;
			   //cerr << (*cb != 0) << "," << (*cc != 0) << "," << (*cd != 0) << endl;
			   //cerr << (*db != 0) << "," << (*dc != 0) << "," << (*dd != 0) << endl << endl;
			}
		}
	}
}

/*
float miniGraph::aStarDist(int strt, int gl){
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


